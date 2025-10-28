#!/usr/bin/env python3
"""
musicViz.py
Generates NGlyph + OGG (Opus) for SebiAI GlyphModder.
Reads zones.json in working directory.
Usage:
    python musicViz.py <audiofile>
"""

import os, sys, json, subprocess
import numpy as np
from scipy.fft import rfft, rfftfreq
from scipy.signal import get_window
from pydub import AudioSegment
import urllib.request
import urllib.error
import time

# ------------------ defaults ------------------
DEFAULTS = {
    "phone_model": "PHONE1",
    "fps": 60,
    "decay_alpha": 0.9,     # smoother falloff
    "title": None,
    # amplifier defaults (overridden by zones.json 'amp' section)
    "amp": {
        "min": 0.5,
        "max": 2.0,
        "initial": 1.0,
        "up_speed": 0.5,    # units per second
        "down_speed": 0.25  # units per second
    }
}

# ------------------ helpers ------------------
def load_audio_mono(path):
    seg = AudioSegment.from_file(path)
    sr = seg.frame_rate
    samples = np.array(seg.get_array_of_samples(), dtype=np.float32)
    if seg.channels == 2:
        samples = samples.reshape((-1, 2)).mean(axis=1)
    samples /= float(2 ** (seg.sample_width * 8 - 1))
    return samples, sr

def ensure_ogg_opus(input_path, out_ogg_path):
    """Export to Ogg Opus using ffmpeg via pydub."""
    a = AudioSegment.from_file(input_path)
    a.export(out_ogg_path, format="ogg", codec="libopus", parameters=["-b:a", "192k"])
    return out_ogg_path

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
        for zi, (low, high) in enumerate(zones):
            raw[i, zi] = compute_zone_peak(spec, freqs, low, high)
        if (i + 1) % tick == 0 or i == n_frames - 1:
            pct = int((i + 1) / n_frames * 100)
            print(f"\r[FFT] {pct}% ({i+1}/{n_frames})", end='', flush=True)
    print()  # newline after progress
    return raw, n_frames

# normalize raw (0..1) to quadratic brightness (0..5000) — kept simple
def normalize_to_linear(raw):
    zone_max = np.max(raw, axis=0)
    zone_max[zone_max == 0] = 1e-12
    scaled = raw / zone_max
    # quadratic mapping for emphasis on peaks (kept as requested)
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
    amp_min = float(amp_conf.get("min", DEFAULTS["amp"]["min"]))
    amp_max = float(amp_conf.get("max", DEFAULTS["amp"]["max"]))
    return float(max(amp_min, min(amp_max, mult)))

# Apply one stable multiplier and perform simple instant-rise / smoothed-fall
def apply_stable_and_smooth(linear, fps, decay_alpha, amp_conf):
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
    fps = conf.get("fps", DEFAULTS["fps"])
    phone_model = conf.get("phone_model", DEFAULTS["phone_model"])
    decay_alpha = conf.get("decay_alpha", DEFAULTS["decay_alpha"])
    zones = conf["zones"]
    amp_conf = conf.get("amp", DEFAULTS["amp"])

    # --- audio analysis
    samples, sr = load_audio_mono(audio_path)

    # compute raw matrix
    raw, n_frames = compute_raw_matrix(samples, sr, zones, fps)
    print(f"[+] sr={sr}, frames={n_frames}")

    # normalize to quadratic linear 0..5000
    linear = normalize_to_linear(raw)

    # apply single-file stable multiplier + smoothing per-frame (realtime-capable)
    final = apply_stable_and_smooth(linear, fps, decay_alpha, amp_conf)

    # --- write NGlyph
    author_rows = [",".join(map(str, row)) + "," for row in final]
    ng = {
        "VERSION": 1,
        "PHONE_MODEL": phone_model,
        "AUTHOR": author_rows,
        "CUSTOM1": ["1-0", "150-1"]
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

    cmd = [sys.executable, glyphmodder_path, "write", "-t", title, arg_nglyph, arg_ogg]
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
            print(f"[+] Downloading GlyphModder from SebiAi's repo (attempt {attempt}) ...")
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

# ------------------ entrypoint ------------------
if __name__ == "__main__":
    # accept optional --update flag to force overwriting GlyphModder.py from GitHub
    update_flag = False
    if "--update" in sys.argv:
        update_flag = True

    # Attempt to pull GlyphModder.py if --update
    if update_flag:
         try:
            download_glyphmodder_to_cwd(overwrite=update_flag)
         except Exception as e_download:
            print(f"[!] Warning: automatic GlyphModder fetch failed: {e_download}")
        # continue — run_glyphmodder_write will still search parent dir / cwd as before

    # Instead of a single input file, process all files in ./input and write to ./output
    input_dir = "Input"
    output_dir = "Output"
    nglyph_dir = "Nglyph"
    cfg_path = "zones.json"

    if not os.path.isdir(input_dir):
        print(f"[+] Creating input directory: {input_dir}")
        os.makedirs(input_dir, exist_ok=True)
        print(f"[!] Please place audio files into the '{input_dir}' folder and re-run.")
        sys.exit(0)

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    if not os.path.isdir(nglyph_dir):
        os.makedirs(nglyph_dir, exist_ok=True)

    if not os.path.isfile(cfg_path):
        print("[!] zones.json not found in working directory.")
        sys.exit(1)

    conf = json.load(open(cfg_path, "r", encoding="utf-8"))
    for k,v in DEFAULTS.items():
        conf.setdefault(k, v)

    files = sorted(os.listdir(input_dir))
    if not files:
        print(f"No files found in '{input_dir}'. Drop audio files there and run again.")
        sys.exit(0)

    processed = []
    for fname in files:
        in_path = os.path.join(input_dir, fname)
        if not os.path.isfile(in_path):
            continue
        base = os.path.splitext(os.path.basename(fname))[0]
        out_nglyph = os.path.join(nglyph_dir, base + ".nglyph")   # save nglyphs under Nglyph/
        # intermediate unmodded ogg (absolute path) and desired final name
        temp_unmod_ogg = os.path.abspath(os.path.join(output_dir, base + ".unmod.ogg"))
        desired_final_ogg = os.path.abspath(os.path.join(output_dir, base + ".ogg"))
        print(f"[+] Processing '{fname}' -> nglyph:'{out_nglyph}' (temp ogg:'{temp_unmod_ogg}') -> final:'{desired_final_ogg}'")
        final_ogg = None
        try:
            # convert/ensure ogg in output folder (writes absolute path)
            ensure_ogg_opus(in_path, temp_unmod_ogg)
            # produce nglyph (writes to Nglyph folder)
            nglyph_path = process(in_path, conf, out_nglyph)
            # attach ogg via GlyphModder; run with cwd=output_dir so final ogg lands in Output
            final_ogg = run_glyphmodder_write(nglyph_path, temp_unmod_ogg, conf.get("title"), cwd=os.path.abspath(output_dir))
            # if GlyphModder produced a differently named file, move it to desired_final_ogg
            try:
                if os.path.abspath(final_ogg) != desired_final_ogg and os.path.isfile(final_ogg):
                    # overwrite if needed
                    try:
                        os.replace(final_ogg, desired_final_ogg)
                    except Exception:
                        # fallback to copy+remove
                        import shutil
                        shutil.copy2(final_ogg, desired_final_gg)
                        os.remove(final_ogg)
            except Exception as e_move:
                print(f"[!] Warning: couldn't rename GlyphModder output: {e_move}")
            processed.append(fname)
        except Exception as e:
            print(f"[!] Failed processing '{fname}': {e}")
        finally:
            # Always attempt to remove the intermediate unmod file(s).
            try:
                if os.path.isfile(temp_unmod_ogg):
                    os.remove(temp_unmod_ogg)
                    print(f"[cleanup] removed intermediate: {temp_unmod_ogg}")
            except Exception:
                pass
            # Also attempt to remove the basename representation inside output_dir (defensive)
            try:
                alt = os.path.join(os.path.abspath(output_dir), os.path.basename(temp_unmod_ogg))
                if os.path.isfile(alt):
                    os.remove(alt)
            except Exception:
                pass

    print(f"[+] Done. Processed {len(processed)} file(s): {processed}. Find them in the output folder!")
