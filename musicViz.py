#!/usr/bin/env python3
"""
musicViz.py - a tool that brings better music visualization to Nothing Phones!
https://github.com/Aleks-Levet/better-nothing-music-visualizer

Copyright (C) 2024  Aleks Levet (aka. SebiAi)

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""

import os, sys, json, subprocess
import numpy as np
from scipy.fft import rfft, rfftfreq
from scipy.signal import get_window
from pydub import AudioSegment
import urllib.request
import urllib.error
import time

# ------------------ helpers ------------------
def convert_to_ogg(input_path, output_path):
    """Convert any audio file to OGG format using pydub."""
    print(f"[+] Converting '{input_path}' to OGG -> '{output_path}'")
    audio = AudioSegment.from_file(input_path)
    audio.export(output_path, format="ogg")
    print(f"[+] Conversion complete: {output_path}")
    return output_path

def load_audio_mono(path):
    seg = AudioSegment.from_file(path)
    sr = seg.frame_rate
    samples = np.array(seg.get_array_of_samples(), dtype=np.float32)
    if seg.channels == 2:
        samples = samples.reshape((-1, 2)).mean(axis=1)
    samples /= float(2 ** (seg.sample_width * 8 - 1))
    return samples, sr

def next_pow2(n):
    p = 1
    while p < n:
        p <<= 1
    return p

def compute_zone_peak(mag, freqs, low, high):
    idx = np.where((freqs >= low) & (freqs <= high))[0]
    return float(np.max(mag[idx])) if idx.size else 0.0

# new helper: compute raw per-frame zone peaks (simpler, with light progress)
def compute_raw_matrix(samples, sr, zones, fps):
    hop = int(round(sr / float(fps)))
    win_len = int(round(sr * 0.025))
    win = get_window("hann", win_len, fftbins=True)
    nfft = next_pow2(win_len)
    freqs = rfftfreq(nfft, 1 / sr)
    n_frames = int(np.ceil(len(samples) / hop))
    print(f"[+] compute_raw_matrix: sr={sr}, hop={hop}, win={win_len}, nfft={nfft}, frames={n_frames}")
    raw = np.zeros((n_frames, len(zones)), dtype=float)

    # light progress: print at roughly 10% increments
    tick = max(1, n_frames // 10)

    for i in range(n_frames):
        start = i * hop
        frame = samples[start:start+win_len]
        if frame.size < win_len:
            frame = np.pad(frame, (0, win_len - frame.size))
        spec = np.abs(rfft(frame * win, n=nfft))

        # accept zone entries like [low, high] or [low, high, "description"]
        for zi, zone in enumerate(zones):
            # robust handling: ensure we have at least two numeric bounds
            if not (isinstance(zone, (list, tuple)) and len(zone) >= 2):
                print(f"[!] Invalid zone entry at index {zi}: {zone!r} -- using 0..0")
                low = 0.0
                high = 0.0
            else:
                try:
                    low = float(zone[0])
                    high = float(zone[1])
                except Exception:
                    print(f"[!] Invalid numeric bounds for zone {zi}: {zone!r} -- using 0..0")
                    low = 0.0
                    high = 0.0

            # swap if bounds reversed (user-supplied reversed ranges were producing empty values)
            if low > high:
                low, high = high, low
                print(f"[!] Warning: swapped zone bounds for zone {zi} -> low={low}, high={high}")

            raw[i, zi] = compute_zone_peak(spec, freqs, low, high)

        if (i + 1) % tick == 0 or i == n_frames - 1:
            pct = int((i + 1) / n_frames * 100)
            print(f"\r[FFT] {pct}% ({i+1}/{n_frames})", end='', flush=True)
    print()  # newline after progress
    return raw, n_frames

# normalize raw (0..1) to quadratic brightness (0..5000)
def normalize_to_quadratic(raw):
    zone_max = np.max(raw, axis=0)
    zone_max[zone_max == 0] = 1e-12
    scaled = raw / zone_max
    # quadratic mapping for emphasis on peaks
    return (np.clip(scaled, 0, 1) ** 2 * 5000.0).astype(float)

# simple stable multiplier: use median of frame maxima
def compute_stable_multiplier(linear, amp_conf):
    if linear.size == 0:
        return 1.0
    frame_maxes = np.max(linear, axis=1)
    pct = float(amp_conf.get("percentile", 50.0))  # default median
    ref = float(np.percentile(frame_maxes, pct))
    target = float(amp_conf.get("target", 3000.0))
    if ref <= 0.0:
        mult = 1.0
    else:
        mult = target / ref
    amp_min = float(amp_conf.get("min"))
    amp_max = float(amp_conf.get("max"))
    return float(max(amp_min, min(amp_max, mult)))

# Apply one stable multiplier and perform simple instant-rise / smoothed-fall
def apply_stable_and_smooth(linear, decay_alpha, amp_conf):
    n_frames, n_zones = linear.shape
    mult = compute_stable_multiplier(linear, amp_conf)
    print(f"[+] stable multiplier: {mult:.3f}")
    linear_scaled = linear * mult

    # simple smoothing: instant rise, exponential-ish fall per-sample
    rows = []
    if n_frames == 0:
        return np.zeros((0, n_zones), dtype=int)
    prev = linear_scaled[0].copy()
    rows.append(np.clip(np.round(prev), 0, 4095).astype(int))

    tick = max(1, n_frames // 10)
    for i in range(1, n_frames):
        cur = linear_scaled[i]
        rise = cur >= prev
        # instant rise
        prev[rise] = cur[rise]
        # smoothed fall
        prev[~rise] = decay_alpha * prev[~rise] + (1.0 - decay_alpha) * cur[~rise]
        rows.append(np.clip(np.round(prev), 0, 4095).astype(int))
        if (i + 1) % tick == 0 or i == n_frames - 1:
            pct = int((i + 1) / n_frames * 100)
            print(f"\r[PROC] {pct}% ({i+1}/{n_frames})", end='', flush=True)
    print()
    return np.vstack(rows)

# ------------------ main processing ------------------
def process(audio_path, conf, out_nglyph_path):
    # --- config
    fps = 60  # FPS is fixed to 60 
    phone_model = conf.get("phone_model")
    decay_alpha = conf.get("decay_alpha")
    zones = conf["zones"]
    amp_conf = conf.get("amp")

    # --- audio analysis
    samples, sr = load_audio_mono(audio_path)

    # compute raw matrix
    raw, n_frames = compute_raw_matrix(samples, sr, zones, fps)
    print(f"[+] sr={sr}, frames={n_frames}")

    # normalize to quadratic linear 0..5000
    linear = normalize_to_quadratic(raw)

    # apply single-file stable multiplier + smoothing per-frame (realtime-capable)
    final = apply_stable_and_smooth(linear, decay_alpha, amp_conf)

    # --- write NGlyph
    author_rows = [",".join(map(str, row)) + "," for row in final]
    ng = {
        "VERSION": 1,
        "PHONE_MODEL": phone_model,
        "AUTHOR": author_rows,
        "CUSTOM1": ["1-0", "1050-1"]
    }

    with open(out_nglyph_path, "w", encoding="utf-8") as f:
        json.dump(ng, f, indent=4)
    print(f"[+] Saved NGlyph: {out_nglyph_path}")
    return out_nglyph_path

def run_glyphmodder_write(nglyph_path, ogg_path, title=None, cwd=None):
    if title is None:
        title = os.path.splitext(os.path.basename(nglyph_path))[0]
    # locate GlyphModder.py in the parent directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    glyphmodder_path = os.path.normpath(os.path.join(script_dir, os.pardir, "GlyphModder.py"))
    # fallback to current working directory if not present in parent
    if not os.path.isfile(glyphmodder_path):
        glyphmodder_path = os.path.normpath(os.path.join(os.getcwd(), "GlyphModder.py"))
    if not os.path.isfile(glyphmodder_path):
        print(f"GlyphModder.py not found in parent directory or working directory. Searched: {glyphmodder_path}")
        print("Downloading GlyphModder.py from SebiAI's GitHub repository...")
        download_glyphmodder_to_cwd(overwrite=1) 
        print(f"[+] Proceeding with the downloaded GlyphModder.py from cwd.")
        
    # ensure NGlyph is an absolute path (so GlyphModder can find it from any cwd)
    arg_nglyph = os.path.abspath(nglyph_path)

    # if we run with cwd set to the output dir, pass only the basename for the ogg
    arg_ogg = os.path.basename(ogg_path) if cwd else ogg_path

    cmd = [sys.executable, glyphmodder_path, "write", "--auto-fix-audio", "-t", title, arg_nglyph, arg_ogg]
    print("[+] Running GlyphModder:", " ".join(cmd), f"(cwd={cwd or os.getcwd()})")
    res = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd)
    if res.returncode != 0:
        print(res.stdout)
        print(res.stderr)
        raise RuntimeError("GlyphModder failed")
    # final ogg should be written into cwd (if provided) or current working dir
    final_ogg_path = os.path.join(cwd or os.getcwd(), arg_ogg) if cwd else os.path.abspath(arg_ogg)
    print(f"[+] GlyphModder produced: {final_ogg_path}")
    return final_ogg_path

# new helper: ensure GlyphModder.py exists in current directory (download from SebiAI if needed)
def download_glyphmodder_to_cwd(overwrite=False, attempts=2, backoff=1.0):
    """
    Try to download GlyphModder.py from SebiAI's GitHub raw URLs into cwd.
    If overwrite is False and a file already exists, do nothing.
    Returns True on success (file now exists in cwd), False otherwise.
    """
    target = os.path.join(os.getcwd(), "GlyphModder.py")
    if os.path.isfile(target) and not overwrite:
        print(f"[+] GlyphModder.py already present in cwd: {target}")
        return True

    url = "https://raw.githubusercontent.com/SebiAi/custom-nothing-glyph-tools/main/GlyphModder.py"
    last_err = None
    for attempt in range(1, attempts + 1):
        try:
            print(f"[+] Downloading GlyphModder from SebiAI's repo (attempt {attempt}) ...")
            with urllib.request.urlopen(url, timeout=10) as resp:
                if resp.status != 200:
                    raise urllib.error.HTTPError(url, resp.status, "Non-200 response", resp.headers, None)
                data = resp.read()
            # write atomically
            tmp = target + ".tmp"
            with open(tmp, "wb") as f:
                f.write(data)
            os.replace(tmp, target)
            print(f"[+] Saved GlyphModder.py -> {target}")
            return True
        except Exception as e:
            last_err = e
            print(f"[!] Download attempt {attempt} failed: {e}")
            time.sleep(backoff * attempt)
    print(f"[!] Failed to download from github: {last_err}")
    print("[!] Could not obtain GlyphModder.py from SebiAI's GitHub.")
    return False

# new helper: generate a short help message from zones.config
def generate_help_from_zones(cfg_path="zones.config"):
    if not os.path.isfile(cfg_path):
        return (f"{cfg_path} not found in working directory.\n"
                "Download it from the repository.")
    try:
        raw = json.load(open(cfg_path, "r", encoding="utf-8"))
    except Exception as e:
        return f"Failed to read {cfg_path}: {e}"
    
    # only include entries that look like phone configs (dicts).  This filters out metadata like decay-alpha.
    conf_map = {k: v for k, v in raw.items() if k != "amp" and isinstance(v, dict)}
    lines = []
    lines.append("Usage: python musicViz.py [--update] [--nglyph] [--np1|--np1s|--np2|--np2a|--np3a]\n")
    lines.append(f"Available configs (from {cfg_path}):")
    for key, cfg in conf_map.items():
        pm = cfg.get("phone_model", "<unknown>")
        desc = cfg.get("description", "")
        zones = cfg.get("zones", []) or []
        lines.append(f"  --{key}: {pm} - {desc} ({len(zones)} zones)")
    lines.append("\nExamples:")
    lines.append("  python musicViz.py --np1          # use np1 config")
    lines.append("  python musicViz.py --np1 --nglyph  # only generate an nglyph file using np1 config")
    lines.append("  python musicViz.py --np1s --update  # update GlyphModder.py and run")
    return "\n".join(lines)

# new helper: validate amp configuration 
def validate_amp_conf(amp_conf):
    if not isinstance(amp_conf, dict):
        raise ValueError("The 'amp' entry must be an object in zones.config (global) or inside a phone config.")
    # require at least min and max to be present
    for key in ("min", "max"):
        if key not in amp_conf:
            raise ValueError(f"amp missing required field '{key}' — please add it to zones.config")
    # Only coerce known numeric keys. Ignore other keys (e.g. description).
    numeric_keys = ("min", "max", "initial", "up_speed", "down_speed", "percentile", "target")
    coerced = {}
    for k in numeric_keys:
        if k in amp_conf:
            v = amp_conf[k]
            try:
                coerced[k] = float(v)
            except Exception:
                raise ValueError(f"amp.{k} must be numeric (got {repr(v)})")
    return coerced

# ------------------ entrypoint ------------------
if __name__ == "__main__":
    # if script called with no args, show help generated from zones.config
    if len(sys.argv) == 1:
        cfg_path_hint = "zones.config"
        print(generate_help_from_zones(cfg_path_hint))
        sys.exit(0)

    # accept optional --update flag to force overwriting GlyphModder.py from GitHub
    update_flag = False
    if "--update" in sys.argv:
        update_flag = True
        # remove flag so it doesn't interfere with other argument handling
        sys.argv = [a for a in sys.argv if a != "--update"]

    # accept --nglyph flag: only produce .nglyph files, skip conversion and GlyphModder
    nglyph_only = False
    if "--nglyph" in sys.argv:
        nglyph_only = True
        sys.argv = [a for a in sys.argv if a != "--nglyph"]
        print("[+] Running in --nglyph mode: will only generate .nglyph files (no audio conversion, no GlyphModder).")

    # determine selected phone config
    selected_phone_key = "np1" # default to "np1" when no --np flag provided
    # look for any CLI args beginning with "--np" (last one wins)
    # ignore the script name at sys.argv[0]
    cli_args = sys.argv[1:]
    if cli_args:
        np_flags = [a for a in cli_args if isinstance(a, str) and a.startswith("--np")]
        if np_flags:
            # map "--np1s" -> "np1s"
            selected_phone_key = np_flags[-1].lstrip("-")

    # Attempt to pull GlyphModder.py into cwd if --update was requested
    if update_flag:
        try:
            download_glyphmodder_to_cwd(overwrite=True)
        except Exception as e_download:
            print(f"[!] Warning: automatic GlyphModder fetch failed: {e_download}")
        # continue — run_glyphmodder_write will still search parent dir / cwd as before
    # prepare directories
    input_dir = "Input"
    output_dir = "Output"
    nglyph_dir = "Nglyph"
    cfg_path = "zones.config"

    if not os.path.isdir(input_dir):
        print(f"[+] Creating input directory: {input_dir}")
        os.makedirs(input_dir, exist_ok=True)
        print(f"[!] Please place audio files into the '{input_dir}' folder and re-run. Supported types include mp3, ogg, and m4a.")
        sys.exit(0)

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    if not os.path.isdir(nglyph_dir):
        os.makedirs(nglyph_dir, exist_ok=True)

    if not os.path.isfile(cfg_path):
        print("[!] zones.config not found in working directory.")
        sys.exit(1)

    raw_cfg = json.load(open(cfg_path, "r", encoding="utf-8"))

    # require multi-config format (no legacy single-config)
    if "zones" in raw_cfg:
        print("[!] Legacy single-config format is no longer supported. Please convert zones.config to the multi-config format (top-level 'amp' and per-phone entries).")
        sys.exit(1)
    # top-level amp must exist
    global_amp = raw_cfg.get("amp")
    if global_amp is None:
        print("[!] zones.config must include a top-level 'amp' object. Please add it.")
        sys.exit(1)

    # read optional global decay value (accept both 'decay-alpha' and 'decay_alpha')
    raw_global_decay = None
    if "decay-alpha" in raw_cfg:
        raw_global_decay = raw_cfg.get("decay-alpha")
    elif "decay_alpha" in raw_cfg:
        raw_global_decay = raw_cfg.get("decay_alpha")
    global_decay = None
    if raw_global_decay is not None:
        try:
            global_decay = float(raw_global_decay)
        except Exception:
            print("[!] Invalid decay value in zones.config; 'decay-alpha' must be numeric.")
            sys.exit(1)
    
    conf_map = {k: v for k, v in raw_cfg.items() if k != "amp"}

    # choose selected config (default to np1 if present, else first key)
    if selected_phone_key is None:
        selected_phone_key = "np1" if "np1" in conf_map else next(iter(conf_map.keys()))

    conf = conf_map.get(selected_phone_key)
    if conf is None:
        print(f"[!] Requested config '{selected_phone_key}' not found in zones.config. Falling back to first available config.")
        conf = conf_map[next(iter(conf_map.keys()))]

    # use per-phone amp if present, otherwise use global amp (do NOT create defaults)
    amp_conf = conf.get("amp") if conf.get("amp") is not None else global_amp
    if amp_conf is None:
        print("[!] Missing 'amp' configuration. Please add a top-level 'amp' in zones.config or an 'amp' object in the selected phone config.")
        sys.exit(1)

    try:
        amp_conf = validate_amp_conf(amp_conf)
    except ValueError as e:
        print(f"[!] Invalid 'amp' configuration: {e}")
        sys.exit(1)

    # inject validated amp back into conf so downstream code reads numeric values
    conf["amp"] = amp_conf

    # validate selected phone config: require 'zones' list and numeric 'decay_alpha'
    if "zones" not in conf or not isinstance(conf["zones"], list):
        print("[!] Selected config missing 'zones' array. Please add a 'zones' list to the phone config in zones.config.")
        sys.exit(1)
    
    # decay-alpha may be present per-phone or provided globally
    if "decay-alpha" not in conf:
        if global_decay is not None:
            conf["decay_alpha"] = global_decay
        else:
            print("[!] Selected config missing 'decay-alpha' and no global 'decay-alpha' provided. Please add one to zones.config.")
            sys.exit(1)
    else:
        try:
            conf["decay_alpha"] = float(conf["decay_alpha"])
        except Exception:
            print("[!] Invalid 'decay_alpha' value in phone config; it must be numeric.")
            sys.exit(1)

    files = sorted(os.listdir(input_dir))
    if not files:
        print(f"No files found in '{input_dir}'. Drop audio files there and run again. Supported types include mp3, ogg, m4a.")
        sys.exit(0)

    processed = 0
    for fname in files:
        in_path = os.path.join(input_dir, fname)
        if not os.path.isfile(in_path):
            continue
        base = os.path.splitext(os.path.basename(fname))[0]
        out_nglyph = os.path.join(nglyph_dir, base + ".nglyph")   # save nglyph file under Nglyph/
        desired_final_ogg = os.path.abspath(os.path.join(output_dir, base + ".ogg"))
        print(f"[+] Processing '{fname}' -> nglyph:'{out_nglyph}' -> final:'{desired_final_ogg}'")
        final_ogg = None
        # produce the nglyph file
        nglyph_path = process(in_path, conf, out_nglyph)
        if not nglyph_only:
            # Convert input audio to OGG in output directory first
            source_ogg = convert_to_ogg(in_path, desired_final_ogg)
            final_ogg = run_glyphmodder_write(nglyph_path, source_ogg, conf.get("title"), cwd=os.path.abspath(output_dir))
            
            # GlyphModder may create _fixed_composed.ogg or _composed.ogg depending on whether audio fix was needed
            # Find the actual composed file and rename it to the clean final name
            output_dir_abs = os.path.abspath(output_dir)
            composed_patterns = [
                os.path.join(output_dir_abs, base + "_fixed_composed.ogg"),
                os.path.join(output_dir_abs, base + "_composed.ogg"),
            ]
            composed_file = None
            for pattern in composed_patterns:
                if os.path.isfile(pattern):
                    composed_file = pattern
                    break
            
            if composed_file:
                # Rename to clean final name
                os.rename(composed_file, desired_final_ogg)
                final_ogg = desired_final_ogg
                print(f"[+] Produced {final_ogg}")
                
                # Clean up intermediate files (_fixed.ogg, original converted ogg if different)
                for suffix in ["_fixed.ogg"]:
                    intermediate = os.path.join(output_dir_abs, base + suffix)
                    if os.path.isfile(intermediate) and intermediate != desired_final_ogg:
                        os.remove(intermediate)
                        print(f"[+] Cleaned up {intermediate}")
            else:
                print(f"[!] Warning: Could not find composed output file for {base}")
        processed += 1
    print("-/-/-/-/-/-/-/-/-/-/-/-/-/-/-")
    print(f"Done! Processed {processed} file(s) in total. Find them in the output folder!")
