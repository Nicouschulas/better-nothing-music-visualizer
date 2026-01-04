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
import urllib.request
import urllib.error
import time
import tempfile
import shutil
import soundfile as sf
from typing import Optional, Tuple
from concurrent.futures import ThreadPoolExecutor  # added typing

# Editable ffmpeg conversion settings (tweak these as needed)
CONVERT_SETTINGS = {
    "ffmpeg_bin": "ffmpeg",  # path to ffmpeg binary
    "quality": 7,            # -q:a for libvorbis (0..10), ignored for libopus
    "bitrate": None,         # e.g. "128k" (used for libopus or if you prefer bitrate over quality)
    "sample_rate": None,     # e.g. 48000 or None to keep source
    "channels": None,        # e.g. 2 or 1 or None to keep source
    "extra_args": []         # additional ffmpeg args list, e.g. ["-vn", "-map_metadata", "-1"]
}

# ================== JETSON ORIN NANO OPTIMIZATIONS ==================
# CuPy for GPU-accelerated FFT (requires: pip install cupy-cuda12x for JetPack 6.x)
try:
    import cupy as cp
    from cupyx.scipy.fft import rfft as cp_rfft
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None

def is_jetson_platform():
    """Detect if running on NVIDIA Jetson device."""
    try:
        with open('/proc/device-tree/model', 'r') as f:
            model = f.read().lower()
            return 'jetson' in model
    except Exception:
        return False

def get_jetson_model():
    """Return Jetson model name or None."""
    try:
        with open('/proc/device-tree/model', 'r') as f:
            return f.read().strip('\x00').strip()
    except Exception:
        return None

# Auto-detect Jetson at module load
IS_JETSON = is_jetson_platform()
if IS_JETSON:
    _model = get_jetson_model()
    print(f"[+] Jetson platform detected: {_model}")
    if CUPY_AVAILABLE:
        try:
            # Try to get device name safely
            dev_id = cp.cuda.Device().id
            # On some Jetson CuPy builds, .name might be missing or bytes
            # cp.cuda.runtime.getDeviceProperties(dev_id) returns a dict
            props = cp.cuda.runtime.getDeviceProperties(dev_id)
            dev_name = props.get('name', b'Unknown GPU').decode('utf-8', 'ignore')
            print(f"[+] CuPy GPU acceleration enabled (CUDA device: {dev_name})")
        except Exception as e:
            print(f"[+] CuPy GPU acceleration enabled (Device detection error: {e})")
    else:
        print("[!] CuPy not installed - FFT will use CPU. Install with: pip install cupy-cuda12x")

# Jetson Orin Nano optimized FFmpeg settings (6-core ARM, no NVENC)
CONVERT_SETTINGS_JETSON = {
    "ffmpeg_bin": "ffmpeg",
    "quality": 7,
    "bitrate": "112k",           # Balanced quality/speed for Opus
    "sample_rate": 48000,        # Opus optimal rate
    "channels": 2,
    "extra_args": [
        "-threads", "6",         # Use all 6 ARM Cortex-A78AE cores
        "-compression_level", "5",  # Lower Opus complexity (default 10, range 0-10)
    ]
}
# =====================================================================

def convert_to_ogg_gstreamer(input_path, output_path, bitrate="112000"):
    """
    Convert audio to OGG/Opus using GStreamer pipeline with hardware decoding (NVDEC) where possible.
    Pipeline: filesrc -> decodebin -> audioconvert -> audioresample -> opusenc -> oggmux -> filesink
    """
    gst_bin = "gst-launch-1.0"
    if subprocess.call(["which", gst_bin], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) != 0:
        raise RuntimeError("gst-launch-1.0 not found")

    # GStreamer pipeline for OGG/Opus conversion
    # decodebin handles hardware decoding if available (NVDEC)
    # opusenc settings: bitrate=112k (balanced), complexity=5 (faster)
    cmd = [
        gst_bin, "-q",
        "filesrc", f"location={input_path}", "!",
        "decodebin", "!",
        "audioconvert", "!",
        "audioresample", "!",
        "opusenc", f"bitrate={bitrate}", "complexity=5", "frame-size=20", "!", 
        "oggmux", "!",
        "filesink", f"location={output_path}"
    ]
    
    print(f"[+] Converting with GStreamer (HW accel): {' '.join(cmd)}")
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"GStreamer conversion failed: {res.stderr}")
    print(f"[+] GStreamer conversion complete: {output_path}")
    return output_path

# ------------------ helpers ------------------
def convert_to_ogg(input_path, output_path):
    """
    Convert any audio file to OGG using ffmpeg with settings from CONVERT_SETTINGS.
    Automatically uses Jetson-optimized settings or GStreamer when detected.
    The function signature is unchanged so existing callers still work.
    """
    # 1. Try GStreamer on Jetson first (Hardware Decoding)
    # DISABLED: GStreamer OGG muxing was leaking metadata/album art, causing "File not created by Composer" errors.
    # Reverting to FFmpeg (with Jetson optimizations) ensures clean files via -map_metadata -1.
    # if IS_JETSON:
    #    try:
    #        # Default to 112k bitrate for GStreamer path
    #        return convert_to_ogg_gstreamer(input_path, output_path, bitrate="112000")
    #    except Exception as e:
    #        print(f"[!] GStreamer conversion failed (falling back to FFmpeg): {e}")

    # 2. Fallback to FFmpeg (Software)
    # Use Jetson settings if on Jetson platform
    settings = CONVERT_SETTINGS_JETSON if IS_JETSON else CONVERT_SETTINGS
    
    ffmpeg = settings.get("ffmpeg_bin", "ffmpeg")
    quality = settings.get("quality")
    bitrate = settings.get("bitrate")
    sr = settings.get("sample_rate")
    channels = settings.get("channels")
    extra = settings.get("extra_args", []) or []

    # -vn: strip video streams (album art/covers that break Nothing Composer)
    # -map_metadata -1: strip source metadata to avoid conflicts
    cmd = [ffmpeg, "-y", "-i", input_path, "-vn", "-map_metadata", "-1"]

    if sr is not None:
        cmd += ["-ar", str(int(sr))]
    if channels is not None:
        cmd += ["-ac", str(int(channels))]

    cmd += ["-c:a", "libopus"]
    if bitrate:
        cmd += ["-b:a", str(bitrate)]
    elif quality is not None:
        # map quality to bitrate fallback if desired; default to 96k..192k mapping could be added
        cmd += ["-b:a", f"{int(quality)*16}k"]
  
    # append any user-specified extra ffmpeg args
    cmd += list(extra)
    cmd += [output_path]
    print(f"[+] Converting with ffmpeg: {' '.join(cmd)}")
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(res.stdout)
        print(res.stderr)
        raise RuntimeError(f"ffmpeg conversion failed for {input_path}")
    print(f"[+] Conversion complete: {output_path}")
    return output_path

def load_audio_gstreamer(path):
    """
    Use GStreamer with hardware acceleration for audio loading on Jetson.
    Uses NVDEC for video containers with audio, or standard decoding.
    Returns (samples, sr) or raises RuntimeError.
    """
    # GStreamer pipeline:
    # filesrc -> decodebin (hw accel) -> audioconvert -> audioresample -> appsink (stdout)
    # We output raw float32 mono audio at 48kHz
    
    # Check if gst-launch-1.0 is available
    gst_bin = "gst-launch-1.0"
    if subprocess.call(["which", gst_bin], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) != 0:
        raise RuntimeError("gst-launch-1.0 not found")

    # Pipeline to decode and dump raw samples to STDOUT
    # audio/x-raw,format=F32LE,rate=48000,channels=1,layout=interleaved
    cmd = [
        gst_bin, "-q", 
        "filesrc", f"location={path}", "!",
        "decodebin", "!",
        "audioconvert", "!",
        "audioresample", "!",
        "audio/x-raw,format=F32LE,rate=48000,channels=1,layout=interleaved", "!",
        "filesink", "location=/dev/stdout"
    ]
    
    # print(f"[+] Running GStreamer pipeline (Zero-Copy): {' '.join(cmd)}")
    try:
        res = subprocess.run(cmd, capture_output=True, check=True)
        # Read raw data directly from stdout buffer
        data = np.frombuffer(res.stdout, dtype=np.float32)
        sr = 48000
        return data, sr
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"GStreamer failed: {e.stderr.decode() if e.stderr else 'Unknown error'}")

def load_audio_mono(path):
    """
    Load audio using soundfile and return (mono_samples, sr).
    Falls back to ffmpeg->wav if soundfile cannot read the source format.
    On Jetson, attempts GStreamer hardware decoding first.
    Samples are float32 in [-1, 1].
    """
    # 1. Try GStreamer on Jetson
    if IS_JETSON:
        try:
            # print("[+] Attempting GStreamer hardware decoding...")
            return load_audio_gstreamer(path)
        except Exception as e:
            print(f"[!] GStreamer decoding failed (falling back to CPU): {e}")

    # 2. Try soundfile (CPU)
    try:
        data, sr = sf.read(path, dtype='float32', always_2d=False)
    except Exception:
        # 3. Fallback to ffmpeg -> wav
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        tmp.close()
        try:
            subprocess.run(
                ["ffmpeg", "-y", "-i", path, tmp.name],
                check=True, capture_output=True, text=True
            )
            data, sr = sf.read(tmp.name, dtype='float32', always_2d=False)
        finally:
            try:
                os.unlink(tmp.name)
            except Exception:
                pass

    # Ensure mono
    if data.ndim == 2 and data.shape[1] > 1:
        samples = data.mean(axis=1).astype(np.float32)
    else:
        samples = data.flatten().astype(np.float32)
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
# Supports GPU acceleration via CuPy on Jetson platforms
def compute_raw_matrix(samples, sr, zones, fps):
    hop = int(round(sr / float(fps)))
    win_len = int(round(sr * 0.025))
    win = get_window("hann", win_len, fftbins=True)
    nfft = next_pow2(win_len)
    freqs = rfftfreq(nfft, 1 / sr)
    n_frames = int(np.ceil(len(samples) / hop))
    
    # Pre-parse zone bounds once
    zone_bounds = []
    for zi, zone in enumerate(zones):
        if not (isinstance(zone, (list, tuple)) and len(zone) >= 2):
            print(f"[!] Invalid zone entry at index {zi}: {zone!r} -- using 0..0")
            zone_bounds.append((0.0, 0.0))
        else:
            try:
                low = float(zone[0])
                high = float(zone[1])
                if low > high:
                    low, high = high, low
                    print(f"[!] Warning: swapped zone bounds for zone {zi} -> low={low}, high={high}")
                zone_bounds.append((low, high))
            except Exception:
                print(f"[!] Invalid numeric bounds for zone {zi}: {zone!r} -- using 0..0")
                zone_bounds.append((0.0, 0.0))
    
    use_gpu = CUPY_AVAILABLE and IS_JETSON
    if use_gpu:
        print(f"[+] compute_raw_matrix [GPU]: sr={sr}, hop={hop}, win={win_len}, nfft={nfft}, frames={n_frames}")
    else:
        print(f"[+] compute_raw_matrix [CPU]: sr={sr}, hop={hop}, win={win_len}, nfft={nfft}, frames={n_frames}")
    
    raw = np.zeros((n_frames, len(zones)), dtype=float)
    tick = max(1, n_frames // 10)

    if use_gpu:
        # GPU-accelerated path using CuPy (Vectorized)
        print(f"[+] compute_raw_matrix [GPU-Vectorized]: Allocating batch matrix ({n_frames}x{win_len})")
        
        # 1. Prepare data on GPU
        # We need to pad the samples to handle the last frame if needed
        total_len = (n_frames - 1) * hop + win_len
        if len(samples) < total_len:
            pad_len = total_len - len(samples)
            samples = np.pad(samples, (0, pad_len))
        
        samples_gpu = cp.asarray(samples, dtype=cp.float32)
        win_gpu = cp.asarray(win, dtype=cp.float32)
        freqs_gpu = cp.asarray(freqs)
        
        # 2. Create strided view (or manual index) to form (n_frames, win_len) matrix
        # Using stride_tricks on GPU can be tricky, let's use simple indexing for now
        # or construct the matrix. Constructing might be memory heavy for very long songs?
        # 8000 frames * 1200 samples * 4 bytes ~= 38MB. Totally fine for Jetson (8GB).
        
        # Create indices: (n_frames, 1) + (1, win_len)
        starts = cp.arange(n_frames) * hop
        indices = starts[:, None] + cp.arange(win_len)[None, :]
        
        # Gather frames: (n_frames, win_len)
        frames_gpu = samples_gpu[indices]
        
        # 3. Apply window: (n_frames, win_len)
        frames_gpu *= win_gpu[None, :]
        
        # 4. Batch FFT: (n_frames, nfft//2 + 1)
        # axis=1 is default for rfft
        spec_gpu = cp.abs(cp_rfft(frames_gpu, n=nfft))
        
        # 5. Compute zone peaks (Vectorized)
        # raw is (n_frames, n_zones)
        raw_gpu = cp.zeros((n_frames, len(zones)), dtype=cp.float32)
        
        for zi, (low, high) in enumerate(zone_bounds):
            # Create mask for this zone: (n_freqs,)
            mask = (freqs_gpu >= low) & (freqs_gpu <= high)
            if cp.any(mask):
                # Max over the frequency axis for this zone
                # spec_gpu[:, mask] selects columns. Then max(axis=1)
                # Note: if mask is empty, max throws error, so we check cp.any
                zone_slice = spec_gpu[:, mask]
                if zone_slice.shape[1] > 0:
                    raw_gpu[:, zi] = cp.max(zone_slice, axis=1)
        
        # 6. Copy back to CPU
        raw = cp.asnumpy(raw_gpu)
        print(f"[+] GPU processing complete.")
        
    else:
        # CPU path (original)
        for i in range(n_frames):
            start = i * hop
            frame = samples[start:start+win_len]
            if frame.size < win_len:
                pad_width: Tuple[int, int] = (0, int(win_len - int(frame.size)))
                frame = np.pad(frame, pad_width)
            spec = np.abs(rfft(frame * win, n=nfft))

            for zi, (low, high) in enumerate(zone_bounds):
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

# new helper: map per-zone brightness using optional zone[3]=low_percent and zone[4]=high_percent
def apply_zone_percent_mapping(linear: np.ndarray, zones, linear_max: float = 5000.0) -> np.ndarray:
    """
    Apply per-zone percent mapping on the pre-smoothed 'linear' matrix.
    - linear: (n_frames, n_zones) float array in range ~0..linear_max (default 5000)
    - zones: list of zone entries where zone[3]=low_percent, zone[4]=high_percent (optional)

    Behaviour:
      perc = (linear_value / linear_max) * 100
      if perc <= low -> mapped_linear = 0
      if perc >= high -> mapped_linear = linear_max
      else -> mapped_linear = ((perc - low)/(high - low)) * linear_max

    Returns mapped linear array in the same scale as 'linear' (float).
    """
    if linear.size == 0:
        return linear
    src = linear.astype(np.float64)
    out = src.copy()
    n_frames, n_zones = out.shape

    def _parse_percent(v):
        try:
            if isinstance(v, str):
                s = v.strip()
                if s.endswith('%'):
                    s = s[:-1].strip()
                val = float(s)
            else:
                val = float(v)
        except Exception:
            return None
        if 0.0 <= val <= 1.0:
            return val * 100.0
        return val

    for zi, zone in enumerate(zones):
        if not (isinstance(zone, (list, tuple)) and len(zone) >= 5):
            continue
        low = _parse_percent(zone[3])
        high = _parse_percent(zone[4])
        if low is None or high is None:
            continue
        low = max(0.0, min(100.0, low))
        high = max(0.0, min(100.0, high))
        if low > high:
            low, high = high, low

        percents = (src[:, zi] / float(linear_max)) * 100.0
        if high == low:
            mask_hi = percents >= high
            out[:, zi] = 0.0
            out[mask_hi, zi] = float(linear_max)
            continue

        below = percents <= low
        above = percents >= high
        between = (~below) & (~above)

        out[below, zi] = 0.0
        out[above, zi] = float(linear_max)
        if np.any(between):
            out[between, zi] = ((percents[between] - low) / (high - low)) * float(linear_max)

    return out

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
 
    # apply per-zone percent mapping (optional zone[3], zone[4]) BEFORE smoothing
     try:
         linear = apply_zone_percent_mapping(linear, zones, linear_max=5000.0)
     except Exception as e:
         print(f"[!] Warning: zone percent mapping failed: {e}")
  
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

def run_glyphmodder_write(nglyph_path: str, ogg_path: str, title: Optional[str] = None, cwd: Optional[str] = None) -> str:
    if not isinstance(ogg_path, str) or not ogg_path:
        raise ValueError("ogg_path must be a non-empty string")
    if title is None:
        title = os.path.splitext(os.path.basename(nglyph_path))[0]
    
    # Append watermark to title (safest way to show it without breaking playback)
    # Using GLYPHER_WATERMARK tag causes "two dots" / unplayable files on phone.
    if "Nite <3" not in title:
        title += " | Nite <3"
    
    # Locate GlyphModder.py
    script_dir = os.path.dirname(os.path.abspath(__file__))
    glyphmodder_path = os.path.normpath(os.path.join(script_dir, os.pardir, "GlyphModder.py"))
    if not os.path.isfile(glyphmodder_path):
        glyphmodder_path = os.path.normpath(os.path.join(os.getcwd(), "GlyphModder.py"))
    
    if not os.path.isfile(glyphmodder_path):
        print(f"GlyphModder.py not found. Downloading...")
        download_glyphmodder_to_cwd(overwrite=True)
        glyphmodder_path = os.path.join(os.getcwd(), "GlyphModder.py")

    # ensure NGlyph is an absolute path
    arg_nglyph = os.path.abspath(nglyph_path)
    arg_ogg = os.path.abspath(ogg_path)
    
    # DIRECT IMPORT INTEGRATION (Optimization)
    try:
        import sys
        import importlib.util
        
        # Load GlyphModder module dynamically
        spec = importlib.util.spec_from_file_location("GlyphModder", glyphmodder_path)
        GlyphModder = importlib.util.module_from_spec(spec)
        sys.modules["GlyphModder"] = GlyphModder
        spec.loader.exec_module(GlyphModder)
        
        print(f"[+] Imported GlyphModder from {glyphmodder_path}")
        
        # Initialize classes
        ffmpeg_helper = GlyphModder.FFmpeg(ffmpeg_path="ffmpeg", ffprobe_path="ffprobe")
        audio_file = GlyphModder.AudioFile(arg_ogg, ffmpeg_helper)
        nglyph_file = GlyphModder.NGlyphFile(arg_nglyph)
        
        output_dir = cwd if cwd else os.path.dirname(arg_ogg)
        
        # Call write function directly
        GlyphModder.write_metadata_to_audio_file(
            audio_file=audio_file,
            nglyph_file=nglyph_file,
            output_path=output_dir,
            title=title,
            ffmpeg=ffmpeg_helper,
            auto_fix_audio=True
        )
        
        # Determine output file path
        # GlyphModder appends '_composed' to the filename
        base_name = os.path.splitext(os.path.basename(arg_ogg))[0]
        possible_out = os.path.join(output_dir, base_name + "_composed.ogg")
        
        # Check for _fixed_composed.ogg as well (if auto-fix happened)
        if not os.path.isfile(possible_out):
             possible_out = os.path.join(output_dir, base_name + "_fixed_composed.ogg")

        # SET ARTIST & ALBUM METADATA (Post-Process)
        # GlyphModder sets ALBUM to "Glyph Tools v2" and doesn't set ARTIST.
        # We force both to "Nite <3" for branding.
        if os.path.isfile(possible_out):
            final_out = os.path.join(output_dir, base_name + "_final.ogg")
            cmd = [
                "ffmpeg", "-y", "-i", possible_out,
                "-metadata", "artist=Nite <3",
                "-metadata", "album=Nite <3",
                "-c", "copy", "-map_metadata", "0",
                final_out
            ]
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if os.path.isfile(final_out):
                os.replace(final_out, possible_out)
                print(f"[+] Set Artist/Album tags to 'Nite <3'")
            
            print(f"[+] GlyphModder (Import) produced: {possible_out}")
            return possible_out
            
        raise RuntimeError(f"Could not find GlyphModder output. Expected: {possible_out}")

    except Exception as e:
        # Fallback to subprocess if import fails
        # print(f"[!] Direct GlyphModder import failed/skipped: {e}")
        
        arg_ogg_base = os.path.basename(ogg_path) if cwd else ogg_path
        cmd = [sys.executable, glyphmodder_path, "write", "--auto-fix-audio", "-t", title, arg_nglyph, arg_ogg_base]
        print("[+] Running GlyphModder (subprocess):", " ".join(cmd), f"(cwd={cwd or os.getcwd()})")
        res = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd)
        if res.returncode != 0:
            print(res.stdout)
            print(res.stderr)
            raise RuntimeError("GlyphModder failed")
        
        final_ogg_path = os.path.join(cwd or os.getcwd(), arg_ogg_base) if cwd else os.path.abspath(arg_ogg_base)
        
        # SET ARTIST METADATA (Post-Process for subprocess path)
        if os.path.isfile(final_ogg_path):
            temp_final = final_ogg_path + ".tmp.ogg"
            cmd = [
                "ffmpeg", "-y", "-i", final_ogg_path,
                "-metadata", "artist=Nite <3",
                "-c", "copy", "-map_metadata", "0",
                temp_final
            ]
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if os.path.isfile(temp_final):
                os.replace(temp_final, final_ogg_path)
                print(f"[+] Set Artist tag to 'Nite <3'")

        print(f"[+] GlyphModder produced: {final_ogg_path}")
        return final_ogg_path

# new helper: ensure GlyphModder.py exists in current directory (download from SebiAI if needed)
def download_glyphmodder_to_cwd(overwrite: bool = False, attempts: int = 2, backoff: float = 1.0) -> bool:
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




# ---------------- api ------------------
# Lightweight, efficient API: removed adaptive zoom and threshold machinery.
# Relies on the standard pipeline: compute_raw_matrix -> normalize_to_quadratic -> apply_stable_and_smooth

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

class GlyphVisualizerAPI:
    """
    Simplified API for generating Nothing Phone glyph visualizer OGG files.
    Uses the main processing pipeline from this module. Adaptive zoom and
    per-frame thresholding removed for simplicity and speed.
    """

    def __init__(self, zones_config_path: str = None):
        if zones_config_path is None:
            zones_config_path = os.path.join(_SCRIPT_DIR, "zones.config")
        if not os.path.isfile(zones_config_path):
            raise FileNotFoundError(f"zones.config not found at: {zones_config_path}")
        self.zones_config_path = zones_config_path
        self._load_config()

    def _load_config(self):
        with open(self.zones_config_path, "r", encoding="utf-8") as f:
            self.raw_config = json.load(f)
        self.global_amp = self.raw_config.get("amp")
        if self.global_amp is None:
            raise ValueError("zones.config must include a top-level 'amp' object")
        raw_decay = self.raw_config.get("decay-alpha") or self.raw_config.get("decay_alpha")
        self.global_decay = float(raw_decay) if raw_decay is not None else None
        self.phone_configs = {
            k: v for k, v in self.raw_config.items()
            if k != "amp" and isinstance(v, dict)
        }

    def get_available_phones(self) -> list:
        return list(self.phone_configs.keys())

    def get_phone_info(self, phone_key: str) -> dict:
        if phone_key not in self.phone_configs:
            raise ValueError(f"Unknown phone key: {phone_key}")
        conf = self.phone_configs[phone_key]
        return {
            "key": phone_key,
            "description": conf.get("description", ""),
            "phone_model": conf.get("phone_model", "UNKNOWN"),
            "zone_count": len(conf.get("zones", []))
        }

    def _prepare_config(self, phone_key: str) -> dict:
        if phone_key not in self.phone_configs:
            raise ValueError(f"Unknown phone key: {phone_key}")
        conf = dict(self.phone_configs[phone_key])
        amp_conf = conf.get("amp") if conf.get("amp") is not None else self.global_amp
        if amp_conf is None:
            raise ValueError("Missing 'amp' configuration")
        conf["amp"] = validate_amp_conf(amp_conf)
        if "decay-alpha" in conf:
            conf["decay_alpha"] = float(conf["decay-alpha"])
        elif "decay_alpha" in conf:
            conf["decay_alpha"] = float(conf["decay_alpha"])
        elif self.global_decay is not None:
            conf["decay_alpha"] = self.global_decay
        else:
            raise ValueError("Missing 'decay_alpha' configuration")
        if "zones" not in conf or not isinstance(conf["zones"], list):
            raise ValueError("Configuration missing 'zones' array")
        return conf

    def _process_audio(self, audio_path: str, conf: dict, out_nglyph_path: str) -> str:
        """
        Process audio using the standard pipeline and write .nglyph.
        This function avoids extra per-frame Python loops for postprocessing.
        """
        fps = 60
        phone_model = conf.get("phone_model")
        decay_alpha = conf.get("decay_alpha")
        zones = conf["zones"]
        amp_conf = conf.get("amp")

        samples, sr = load_audio_mono(audio_path)
        raw, n_frames = compute_raw_matrix(samples, sr, zones, fps)

        # Vectorized normalization + smoothing pipeline
        linear = normalize_to_quadratic(raw)  # float matrix
        try:
            linear = apply_zone_percent_mapping(linear, zones, linear_max=5000.0)
        except Exception as e:
            print(f"[!] Warning: zone percent mapping failed in API: {e}")
        final = apply_stable_and_smooth(linear, decay_alpha, amp_conf)  # int matrix

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
        return out_nglyph_path

    def generate_glyph_ogg(
        self,
        audio_path: str,
        phone_model: str = "np1",
        output_path: str = None,
        title: str = None,
        nglyph_only: bool = False
    ) -> str:
        """
        Generate glyph OGG (or .nglyph if nglyph_only=True).
        Simpler call signature and faster internals than the previous API.
        """
        if not os.path.isfile(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        conf = self._prepare_config(phone_model)
        base_name = os.path.splitext(os.path.basename(audio_path))[0]
        if title is None:
            title = base_name

        work_dir = tempfile.mkdtemp(prefix="glyph_viz_")
        try:
            nglyph_path = os.path.join(work_dir, f"{base_name}.nglyph")
            ogg_path = os.path.join(work_dir, f"{base_name}.ogg")
            
            # PARALLEL EXECUTION OPTIMIZATION
            # Run FFT analysis (GPU) and OGG conversion (CPU/NVDEC) in parallel.
            print(f"[+] Starting Parallel Execution: FFT (GPU) + Encoding (CPU/NVDEC)")
            with ThreadPoolExecutor(max_workers=2) as executor:
                future_nglyph = executor.submit(self._process_audio, audio_path, conf, nglyph_path)
                
                if not nglyph_only:
                    future_ogg = executor.submit(convert_to_ogg, audio_path, ogg_path)
                
                # Wait for FFT results
                future_nglyph.result()
                
                if nglyph_only:
                    if output_path is None:
                        output_path = os.path.join(os.path.dirname(audio_path), f"{base_name}.nglyph")
                    shutil.copy2(nglyph_path, output_path)
                    return output_path
                
                # Wait for OGG conversion
                future_ogg.result()

            # Ensure GlyphModder is available in work_dir
            original_cwd = os.getcwd()
            os.chdir(work_dir)
            try:
                download_glyphmodder_to_cwd(overwrite=False)
            finally:
                os.chdir(original_cwd)

            run_glyphmodder_write(nglyph_path, ogg_path, title=title, cwd=work_dir)

            # Find produced file and copy to output_path
            composed_patterns = [
                os.path.join(work_dir, f"{base_name}_fixed_composed.ogg"),
                os.path.join(work_dir, f"{base_name}_composed.ogg"),
            ]
            composed_file = next((p for p in composed_patterns if os.path.isfile(p)), None)
            if not composed_file:
                raise RuntimeError(f"GlyphModder did not produce expected output. Check work_dir: {work_dir}")

            if output_path is None:
                output_path = os.path.join(os.path.dirname(audio_path), f"{base_name}_glyph.ogg")
            shutil.copy2(composed_file, output_path)
            return output_path
        finally:
            try:
                shutil.rmtree(work_dir)
            except Exception:
                pass

# Convenience wrapper (keeps previous simple-call behavior)
def generate_glyph_ogg(
    audio_path: str,
    phone_model: str = "np1",
    output_path: str = None,
    title: str = None
) -> str:
    api = GlyphVisualizerAPI()
    return api.generate_glyph_ogg(audio_path, phone_model, output_path, title)
#-------------------api end-------------------




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
