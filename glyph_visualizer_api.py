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
    
    def generate_glyph_ogg(
        self,
        audio_path: str,
        phone_model: str = "np1",
        output_path: str = None,
        title: str = None,
        nglyph_only: bool = False,
        cutoff_threshold: int = 0,
        cutoff_trigger: int = 100
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
            
            # Generate NGlyph with cutoff parameters
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
