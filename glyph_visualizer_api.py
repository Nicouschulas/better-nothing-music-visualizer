#!/usr/bin/env python3
"""
glyph_visualizer_api.py - API wrapper for musicViz.py

Provides a clean, importable API for generating Nothing Phone glyph visualizer
OGG files from audio inputs.
"""

import os
import sys
import json
import tempfile
import shutil

# Add the visualizer directory to path for imports
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

# Import functions from musicViz
from musicViz import (
    convert_to_ogg,
    load_audio_mono,
    compute_raw_matrix,
    normalize_to_quadratic,
    apply_stable_and_smooth,
    validate_amp_conf,
    run_glyphmodder_write,
    download_glyphmodder_to_cwd
)

import numpy as np
from scipy.fft import rfft, rfftfreq
from scipy.signal import get_window


# ================== ADAPTIVE ZOOM HELPERS ==================

def compute_band_energy(samples, sr, fps, buffer_seconds=2.0):
    """
    Compute energy in 3 frequency bands over time with rolling buffer.
    
    Bands:
        - Low: 20-200Hz (bass/sub)
        - Mid: 200-2000Hz (vocals/instruments)
        - High: 2000-16000Hz (treble/air)
    
    Returns:
        band_energy: (n_frames, 3) array with [low, mid, high] energy per frame
        dominant_band: (n_frames,) array with dominant band index per frame (-1 if none >70%)
        dominance_ratio: (n_frames,) ratio of dominant band energy
    """
    hop = int(round(sr / float(fps)))
    win_len = int(round(sr * 0.025))
    win = get_window("hann", win_len, fftbins=True)
    nfft = 2 ** int(np.ceil(np.log2(win_len)))
    freqs = rfftfreq(nfft, 1 / sr)
    
    n_frames = int(np.ceil(len(samples) / hop))
    buffer_frames = int(buffer_seconds * fps)
    
    # Frequency band masks
    low_mask = (freqs >= 20) & (freqs < 200)
    mid_mask = (freqs >= 200) & (freqs < 2000)
    high_mask = (freqs >= 2000) & (freqs < 16000)
    
    band_energy = np.zeros((n_frames, 3))
    
    for i in range(n_frames):
        start = i * hop
        frame = samples[start:start + win_len]
        if frame.size < win_len:
            frame = np.pad(frame, (0, win_len - frame.size))
        
        spec = np.abs(rfft(frame * win, n=nfft)) ** 2  # Power spectrum
        
        band_energy[i, 0] = np.sum(spec[low_mask])
        band_energy[i, 1] = np.sum(spec[mid_mask])
        band_energy[i, 2] = np.sum(spec[high_mask])
    
    # Compute rolling average for smoother detection
    smoothed_energy = np.zeros_like(band_energy)
    for i in range(n_frames):
        start_idx = max(0, i - buffer_frames + 1)
        smoothed_energy[i] = np.mean(band_energy[start_idx:i+1], axis=0)
    
    # Determine dominant band
    total_energy = np.sum(smoothed_energy, axis=1, keepdims=True)
    total_energy[total_energy == 0] = 1e-12  # Avoid division by zero
    ratios = smoothed_energy / total_energy
    
    dominant_band = np.full(n_frames, -1, dtype=int)
    dominance_ratio = np.zeros(n_frames)
    
    for i in range(n_frames):
        max_ratio = np.max(ratios[i])
        if max_ratio >= 0.70:  # 70% threshold
            dominant_band[i] = np.argmax(ratios[i])
            dominance_ratio[i] = max_ratio
    
    return band_energy, dominant_band, dominance_ratio


def generate_zoomed_zones(base_zones, target_range, num_progress_zones=24, progress_zone_start=0):
    """
    Generate new frequency zones focused on a target range.
    
    Args:
        base_zones: Original zone list
        target_range: (low_hz, high_hz) to zoom into
        num_progress_zones: Number of zones in the progress bar
        progress_zone_start: Index where progress zones start in base_zones
    
    Returns:
        New zones list with progress zones remapped to target range
    """
    zoomed = list(base_zones)  # Copy original
    low_hz, high_hz = target_range
    
    # Logarithmic distribution within the zoomed range
    log_low = np.log10(max(20, low_hz))
    log_high = np.log10(min(20000, high_hz))
    log_freqs = np.logspace(log_low, log_high, num_progress_zones + 1)
    
    # Replace only the progress zone portion
    for i in range(num_progress_zones):
        zone_idx = progress_zone_start + i
        if zone_idx >= len(zoomed):
            break
        zone_low = log_freqs[i]
        zone_high = log_freqs[i + 1]
        # Preserve original description
        desc = zoomed[zone_idx][2] if len(zoomed[zone_idx]) > 2 else f"Zone {zone_idx+1}"
        zoomed[zone_idx] = [zone_low, zone_high, desc]
    
    return zoomed


def interpolate_zones(zones_from, zones_to, alpha, num_progress_zones=24, progress_zone_start=0):
    """
    Smoothly interpolate between two zone configurations.
    
    Args:
        zones_from: Starting zones
        zones_to: Target zones
        alpha: Interpolation factor (0=from, 1=to)
        num_progress_zones: Number of progress zones to interpolate
        progress_zone_start: Index where progress zones start
    
    Returns:
        Interpolated zones list
    """
    result = list(zones_from)  # Copy original
    
    for i in range(num_progress_zones):
        zone_idx = progress_zone_start + i
        if zone_idx >= len(zones_from) or zone_idx >= len(zones_to):
            break
        
        low = zones_from[zone_idx][0] * (1 - alpha) + zones_to[zone_idx][0] * alpha
        high = zones_from[zone_idx][1] * (1 - alpha) + zones_to[zone_idx][1] * alpha
        desc = zones_from[zone_idx][2] if len(zones_from[zone_idx]) > 2 else f"Zone {zone_idx+1}"
        result[zone_idx] = [low, high, desc]
    
    return result


def apply_adaptive_zoom(samples, sr, base_zones, fps=60, 
                        transition_time=0.5, num_progress_zones=24, progress_zone_start=0):
    """
    Apply adaptive spectrum zoom based on dominant frequency detection.
    
    Args:
        samples: Audio samples
        sr: Sample rate
        base_zones: Original zone configuration
        fps: Frames per second
        transition_time: Seconds to transition between zoom levels
        num_progress_zones: Number of zones in progress bar
        progress_zone_start: Index where progress zones start in base_zones
    
    Returns:
        List of (frame_zones, transition_alpha) for each frame
    """
    n_frames = int(np.ceil(len(samples) / int(round(sr / float(fps)))))
    transition_frames = int(transition_time * fps)
    
    # Compute band energy and dominance
    _, dominant_band, dominance_ratio = compute_band_energy(samples, sr, fps)
    
    # Define zoom ranges for each band
    zoom_ranges = {
        0: (20, 300),      # Bass zoom: 20-300Hz
        1: (150, 3000),    # Mid zoom: 150-3000Hz  
        2: (1500, 16000),  # High zoom: 1500-16000Hz
        -1: (20, 16000),   # Full range (no zoom)
    }
    
    # Track current zoom state
    current_zoom = -1
    zoom_progress = 0.0  # 0 = fully at base, 1 = fully zoomed
    target_zoom = -1
    
    frame_zones = []
    
    for i in range(n_frames):
        new_dominant = dominant_band[i]
        
        # Update target if dominance changed
        if new_dominant != target_zoom:
            target_zoom = new_dominant
            
        # Smoothly transition zoom_progress
        if target_zoom != -1 and current_zoom == -1:
            # Zooming in
            zoom_progress = min(1.0, zoom_progress + 1.0 / transition_frames)
            current_zoom = target_zoom
        elif target_zoom == -1 and current_zoom != -1:
            # Zooming out
            zoom_progress = max(0.0, zoom_progress - 1.0 / transition_frames)
            if zoom_progress <= 0:
                current_zoom = -1
        elif target_zoom != current_zoom and target_zoom != -1:
            # Switching between bands
            zoom_progress = min(1.0, zoom_progress + 1.0 / transition_frames)
            if zoom_progress >= 1.0:
                current_zoom = target_zoom
                zoom_progress = 1.0
        
        # Generate zones for this frame
        if current_zoom == -1 or zoom_progress <= 0:
            frame_zones.append(base_zones)
        else:
            zoomed = generate_zoomed_zones(
                base_zones, zoom_ranges[current_zoom], 
                num_progress_zones, progress_zone_start
            )
            interpolated = interpolate_zones(
                base_zones, zoomed, zoom_progress, 
                num_progress_zones, progress_zone_start
            )
            frame_zones.append(interpolated)
    
    return frame_zones


class GlyphVisualizerAPI:
    """
    API for generating Nothing Phone glyph visualizer OGG files.
    
    Usage:
        api = GlyphVisualizerAPI()
        output_path = api.generate_glyph_ogg("input.mp3", phone_model="np1")
    """
    
    def __init__(self, zones_config_path: str = None):
        """
        Initialize the Glyph Visualizer API.
        
        Args:
            zones_config_path: Path to zones.config file. Defaults to the one
                              in the same directory as this module.
        """
        if zones_config_path is None:
            zones_config_path = os.path.join(_SCRIPT_DIR, "zones.config")
        
        if not os.path.isfile(zones_config_path):
            raise FileNotFoundError(f"zones.config not found at: {zones_config_path}")
        
        self.zones_config_path = zones_config_path
        self._load_config()
    
    def _load_config(self):
        """Load and parse the zones configuration."""
        with open(self.zones_config_path, "r", encoding="utf-8") as f:
            self.raw_config = json.load(f)
        
        # Extract global settings
        self.global_amp = self.raw_config.get("amp")
        if self.global_amp is None:
            raise ValueError("zones.config must include a top-level 'amp' object")
        
        # Extract global decay
        self.global_decay = None
        raw_decay = self.raw_config.get("decay-alpha") or self.raw_config.get("decay_alpha")
        if raw_decay is not None:
            self.global_decay = float(raw_decay)
        
        # Extract phone configurations
        self.phone_configs = {
            k: v for k, v in self.raw_config.items() 
            if k not in ("amp", "decay-alpha", "decay_alpha", "what-is-decay-alpha", "what-is-decay")
            and isinstance(v, dict)
        }
    
    def get_available_phones(self) -> list:
        """
        Get list of available phone model keys.
        
        Returns:
            List of phone model keys (e.g., ['np1', 'np1s', 'np2', 'np2a', 'np3a'])
        """
        return list(self.phone_configs.keys())
    
    def get_phone_info(self, phone_key: str) -> dict:
        """
        Get information about a specific phone configuration.
        
        Args:
            phone_key: Phone model key (e.g., 'np1', 'np2a')
            
        Returns:
            Dict with 'description', 'phone_model', and 'zone_count'
        """
        if phone_key not in self.phone_configs:
            raise ValueError(f"Unknown phone key: {phone_key}. Available: {self.get_available_phones()}")
        
        conf = self.phone_configs[phone_key]
        return {
            "key": phone_key,
            "description": conf.get("description", ""),
            "phone_model": conf.get("phone_model", "UNKNOWN"),
            "zone_count": len(conf.get("zones", []))
        }
    
    def _prepare_config(self, phone_key: str) -> dict:
        """Prepare a complete configuration dict for processing."""
        if phone_key not in self.phone_configs:
            raise ValueError(f"Unknown phone key: {phone_key}. Available: {self.get_available_phones()}")
        
        conf = dict(self.phone_configs[phone_key])
        
        # Validate and set amp
        amp_conf = conf.get("amp") if conf.get("amp") is not None else self.global_amp
        if amp_conf is None:
            raise ValueError("Missing 'amp' configuration")
        conf["amp"] = validate_amp_conf(amp_conf)
        
        # Validate and set decay_alpha
        if "decay-alpha" in conf:
            conf["decay_alpha"] = float(conf["decay-alpha"])
        elif "decay_alpha" in conf:
            conf["decay_alpha"] = float(conf["decay_alpha"])
        elif self.global_decay is not None:
            conf["decay_alpha"] = self.global_decay
        else:
            raise ValueError("Missing 'decay_alpha' configuration")
        
        # Validate zones
        if "zones" not in conf or not isinstance(conf["zones"], list):
            raise ValueError("Configuration missing 'zones' array")
        
        return conf
    
    def _process_audio(
        self, 
        audio_path: str, 
        conf: dict, 
        out_nglyph_path: str,
        cutoff_threshold: int = 0,
        cutoff_trigger: int = 100
    ) -> str:
        """
        Process audio file and generate NGlyph file.
        
        Args:
            audio_path: Path to input audio file
            conf: Prepared configuration dict
            out_nglyph_path: Path for output .nglyph file
            cutoff_threshold: Brightness values below this are cut to 0 when
                             at least one glyph exceeds cutoff_trigger. Default 0 (disabled).
            cutoff_trigger: Minimum brightness required to trigger cutoff. Default 100.
            
        Returns:
            Path to generated .nglyph file
        """
        fps = 60
        phone_model = conf.get("phone_model")
        decay_alpha = conf.get("decay_alpha")
        zones = conf["zones"]
        amp_conf = conf.get("amp")
        
        # Load and analyze audio
        samples, sr = load_audio_mono(audio_path)
        raw, n_frames = compute_raw_matrix(samples, sr, zones, fps)
        
        # Normalize and smooth
        linear = normalize_to_quadratic(raw)
        final = apply_stable_and_smooth(linear, decay_alpha, amp_conf)
        
        # Apply brightness cutoff if enabled (cutoff_threshold > 0)
        if cutoff_threshold > 0:
            import numpy as np
            final_array = np.array(final)
            for i in range(len(final_array)):
                row = final_array[i]
                # If any glyph exceeds the trigger threshold
                if np.max(row) >= cutoff_trigger:
                    # Zero out glyphs below the cutoff threshold
                    row[row < cutoff_threshold] = 0
                    final_array[i] = row
            final = final_array.tolist()
        
        # Write NGlyph file
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
    
    def _process_audio_adaptive(
        self, 
        audio_path: str, 
        conf: dict, 
        out_nglyph_path: str,
        cutoff_threshold: int = 0,
        cutoff_trigger: int = 100,
        num_progress_zones: int = 24,
        progress_zone_start: int = 0,
        transition_time: float = 0.5
    ) -> str:
        """
        Process audio with adaptive spectrum zoom.
        
        When a frequency band is dominant (>70% energy), the progress bar zones
        smoothly zoom to focus on that frequency range, showing more detail.
        
        Args:
            audio_path: Path to input audio file
            conf: Prepared configuration dict
            out_nglyph_path: Path for output .nglyph file
            cutoff_threshold: Brightness cutoff (0=disabled)
            cutoff_trigger: Trigger for cutoff
            num_progress_zones: Number of zones in progress bar
            progress_zone_start: Index where progress zones start
            transition_time: Seconds to transition between zoom levels
            
        Returns:
            Path to generated .nglyph file
        """
        from musicViz import next_pow2, compute_zone_peak
        
        fps = 60
        phone_model = conf.get("phone_model")
        decay_alpha = conf.get("decay_alpha")
        base_zones = conf["zones"]
        amp_conf = conf.get("amp")
        
        # Load audio
        samples, sr = load_audio_mono(audio_path)
        
        print("[+] Computing adaptive spectrum zoom...")
        
        # Get per-frame zone configurations with adaptive zoom
        frame_zones = apply_adaptive_zoom(
            samples, sr, base_zones, fps, 
            transition_time, num_progress_zones, progress_zone_start
        )
        
        n_frames = len(frame_zones)
        n_zones = len(base_zones)
        
        # Process each frame with its specific zone configuration
        hop = int(round(sr / float(fps)))
        win_len = int(round(sr * 0.025))
        win = get_window("hann", win_len, fftbins=True)
        nfft = next_pow2(win_len)
        freqs = rfftfreq(nfft, 1 / sr)
        
        raw = np.zeros((n_frames, n_zones), dtype=float)
        
        tick = max(1, n_frames // 10)
        for i in range(n_frames):
            start = i * hop
            frame = samples[start:start + win_len]
            if frame.size < win_len:
                frame = np.pad(frame, (0, win_len - frame.size))
            
            spec = np.abs(rfft(frame * win, n=nfft))
            zones_for_frame = frame_zones[i]
            
            for zi, zone in enumerate(zones_for_frame):
                if zi >= n_zones:
                    break
                if not (isinstance(zone, (list, tuple)) and len(zone) >= 2):
                    raw[i, zi] = 0.0
                    continue
                    
                low = float(zone[0])
                high = float(zone[1])
                if low > high:
                    low, high = high, low
                
                raw[i, zi] = compute_zone_peak(spec, freqs, low, high)
            
            if (i + 1) % tick == 0 or i == n_frames - 1:
                pct = int((i + 1) / n_frames * 100)
                print(f"\r[ADAPTIVE] {pct}% ({i+1}/{n_frames})", end='', flush=True)
        
        print()
        
        # Normalize and smooth using standard pipeline
        linear = normalize_to_quadratic(raw)
        final = apply_stable_and_smooth(linear, decay_alpha, amp_conf)
        
        # Apply brightness cutoff if enabled
        if cutoff_threshold > 0:
            final_array = np.array(final)
            for i in range(len(final_array)):
                row = final_array[i]
                if np.max(row) >= cutoff_trigger:
                    row[row < cutoff_threshold] = 0
                    final_array[i] = row
            final = final_array.tolist()
        
        # Write NGlyph file
        author_rows = [",".join(map(str, row)) + "," for row in final]
        ng = {
            "VERSION": 1,
            "PHONE_MODEL": phone_model,
            "AUTHOR": author_rows,
            "CUSTOM1": ["1-0", "1050-1"]
        }
        
        with open(out_nglyph_path, "w", encoding="utf-8") as f:
            json.dump(ng, f, indent=4)
        
        print(f"[+] Saved adaptive NGlyph: {out_nglyph_path}")
        return out_nglyph_path
    
    def generate_glyph_ogg(
        self,
        audio_path: str,
        phone_model: str = "np1",
        output_path: str = None,
        title: str = None,
        nglyph_only: bool = False,
        cutoff_threshold: int = 0,
        cutoff_trigger: int = 100,
        adaptive_zoom: bool = False
    ) -> str:
        """
        Generate a glyph visualizer OGG file from an audio file.
        
        Args:
            audio_path: Path to input audio file (mp3, m4a, ogg, wav, etc.)
            phone_model: Phone model key (default: 'np1'). Use get_available_phones()
                        to see options.
            output_path: Path for output OGG file. If None, uses temp directory
                        with same base name as input.
            title: Title for the glyph composition. Defaults to input filename.
            nglyph_only: If True, only generate .nglyph file (skip GlyphModder)
            cutoff_threshold: Brightness values below this are cut to 0 when
                             at least one glyph exceeds cutoff_trigger. Default 0 (disabled).
            cutoff_trigger: Minimum brightness required to trigger cutoff. Default 100.
            adaptive_zoom: If True, dynamically zoom spectrum to focus on dominant
                          frequency bands. Creates a more dynamic visualization.
            
        Returns:
            Path to the generated OGG file (or .nglyph if nglyph_only=True)
        """
        if not os.path.isfile(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        # Prepare configuration
        conf = self._prepare_config(phone_model)
        
        # Set up paths
        base_name = os.path.splitext(os.path.basename(audio_path))[0]
        if title is None:
            title = base_name
        
        # Create temp directory for intermediate files
        work_dir = tempfile.mkdtemp(prefix="glyph_viz_")
        
        try:
            nglyph_path = os.path.join(work_dir, f"{base_name}.nglyph")
            
            # Generate NGlyph - use adaptive processing if enabled
            if adaptive_zoom:
                # Determine progress zones based on phone model structure
                # Each phone has a different "progress bar" layout:
                # - np1: zones 7-14 (8 zones) are battery bar progress
                # - np2: zones 8-15 (8 zones) are battery bar progress  
                # - np2a: zones 0-23 (24 zones) are the main spectrum
                # - np3a: zones 0-35 (36 zones) are ALL spectrum (entire display is progress bar)
                # - np1s: only 5 zones, no real progress bar (zones 2-4)
                
                # (num_zones, start_index) for each phone
                progress_config = {
                    "np1":  (8, 7),    # Battery bar: zones 7-14
                    "np1s": (3, 2),    # Simplified: zones 2-4
                    "np2":  (8, 8),    # Battery bar: zones 8-15
                    "np2a": (24, 0),   # Main spectrum: zones 0-23
                    "np3a": (36, 0),   # Entire display: zones 0-35
                }
                num_progress_zones, progress_zone_start = progress_config.get(
                    phone_model, (len(conf["zones"]), 0)
                )
                
                self._process_audio_adaptive(
                    audio_path, conf, nglyph_path,
                    cutoff_threshold=cutoff_threshold,
                    cutoff_trigger=cutoff_trigger,
                    num_progress_zones=num_progress_zones,
                    progress_zone_start=progress_zone_start,
                    transition_time=0.5
                )
            else:
                self._process_audio(
                    audio_path, conf, nglyph_path,
                    cutoff_threshold=cutoff_threshold,
                    cutoff_trigger=cutoff_trigger
                )
            
            if nglyph_only:
                # Copy nglyph to output location
                if output_path is None:
                    output_path = os.path.join(os.path.dirname(audio_path), f"{base_name}.nglyph")
                shutil.copy2(nglyph_path, output_path)
                return output_path
            
            # Convert to OGG and run GlyphModder
            ogg_path = os.path.join(work_dir, f"{base_name}.ogg")
            convert_to_ogg(audio_path, ogg_path)
            
            # Ensure GlyphModder is available
            original_cwd = os.getcwd()
            os.chdir(work_dir)
            try:
                download_glyphmodder_to_cwd(overwrite=False)
            finally:
                os.chdir(original_cwd)
            
            # Run GlyphModder
            run_glyphmodder_write(nglyph_path, ogg_path, title=title, cwd=work_dir)
            
            # Find the composed output file
            composed_patterns = [
                os.path.join(work_dir, f"{base_name}_fixed_composed.ogg"),
                os.path.join(work_dir, f"{base_name}_composed.ogg"),
            ]
            composed_file = None
            for pattern in composed_patterns:
                if os.path.isfile(pattern):
                    composed_file = pattern
                    break
            
            if not composed_file:
                raise RuntimeError(f"GlyphModder did not produce expected output. Check work_dir: {work_dir}")
            
            # Copy to final output location
            if output_path is None:
                output_path = os.path.join(os.path.dirname(audio_path), f"{base_name}_glyph.ogg")
            
            shutil.copy2(composed_file, output_path)
            return output_path
            
        finally:
            # Clean up temp directory
            try:
                shutil.rmtree(work_dir)
            except Exception:
                pass  # Best effort cleanup


# Convenience function for quick use
def generate_glyph_ogg(
    audio_path: str,
    phone_model: str = "np1",
    output_path: str = None,
    title: str = None
) -> str:
    """
    Convenience function to generate glyph OGG in one call.
    
    See GlyphVisualizerAPI.generate_glyph_ogg() for full documentation.
    """
    api = GlyphVisualizerAPI()
    return api.generate_glyph_ogg(audio_path, phone_model, output_path, title)


# Test if run directly
if __name__ == "__main__":
    api = GlyphVisualizerAPI()
    print("Available phone models:")
    for phone_key in api.get_available_phones():
        info = api.get_phone_info(phone_key)
        print(f"  {phone_key}: {info['description']} ({info['zone_count']} zones)")
