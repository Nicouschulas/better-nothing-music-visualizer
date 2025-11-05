# Better-nothing-music-visualizer
This script generates **NGlyph** light animations from any audio file, then runs the generated file through *SebiAi's GlyphModder* to have **better music visualization on nothing phones**. Because the one that Nothing made is *not that good* after all.

# This repo is still under construction! Feel free to contribute!

## Overview

`musicViz.py` takes an audio file (like `.mp3`, `.m4a`, `.ogg`) and creates a **NGlyph** file containing the Glyph animations, and makes and an  glyphed **OGG** file for playback in *Glyph composer* or in *Glyphify*.


## How it works:

* **FFT (Fast Fourier Transform)** is used to analyze frequencies in a 20ms window for each 16.666ms frame (60fps), making the visualization more accurate.
* **Frequency ranges** can be defined in `zones.config` and can be fully customized to your liking.
* The **brightness** of each glyph is defined by the peak magnitude found in the frequancy range defined in `zones.config`. It measures how loud different frequency ranges (“zones”) are.
* **Downward-only smoothing** is applied to make it look smoother while preserving reactiveness.
* A `.nglyph` file is then generated with all the brightness data (see the [NGlyph Format](#-the-nglyph-format)).
*  **SebiAi's** `GlyphModder.py` then converts the `.nglyph` file to a glyphed `.ogg` *ringtone* playable on **Nothing Phones**, containing the audio and the glyph animation.

---

## How to use `MusicViz.py`

## First time setup

Make sure you have **Python** and **ffmpeg** installed.

Then install the required Python packages:

```bash
pip install numpy scipy pydub
```
You can also install the dependencies with the `requirements.txt` file:
```bash
pip install -r requirements.txt
```

## Role of each file

```
Input/       → Drop your audio files here.
Output/      → Final .ogg compositions appear here.
Nglyph/      → Generated .nglyph files are saved here.
zones.config → Required configuration file.
```

---

## ⚙️ zones.config

`zones.config` defines how frequencies map to phone LED zones and amplitude settings.

It can contain multiple phone models, like:

```json
{
  "amp": { "min": 0.5, "max": 2.0, "target": 3000 },
  "np1": {
    "phone_model": "PHONE1",
    "decay-alpha": 0.9,
    "zones": [[50, 120], [120, 300], [300, 600], [600, 1200]]
  },
  "np2a": {
    "phone_model": "PHONE2A",
    "decay-alpha": 0.88,
    "zones": [[60, 150], [150, 400], [400, 1000], [1000, 2000]]
  }
}
```

---

## ▶️ Usage

### Normal

```bash
python musicViz.py --np1
```

Processes all files in `Input/` using the `np1` config.

### Help

```bash
python musicViz.py
```

If no arguments are given, it shows available phone configs listed in 
`zones.config`.

### Update GlyphModder automatically

```bash
python musicViz.py --update
```

Downloads the latest **GlyphModder.py** from SebiAI’s GitHub repo if missing or outdated.

### Generate NGlyph only (no OGG and GlyphModder)

```bash
python musicViz.py --nglyph
```

Only produces `.nglyph` files inside `Nglyph/` and skips `GlyphModder.py`.

---

## The NGlyph Format

The output `.nglyph` file is a JSON file containing:

* `VERSION`: format version (currently `1`)
* `PHONE_MODEL`: which Nothing phone it’s for (`PHONE1`, `PHONE2A`, etc.)
* `AUTHOR`: the frame-by-frame light intensity data
* `CUSTOM1`: metadata for the dots in the *Glyph Composer* app.

Example:

```json
{
    "VERSION": 1,
    "PHONE_MODEL": "PHONE2A",
    "AUTHOR": [
        "0,0,0,0,4095,4095,0,",
        "0,0,0,4095,4095,4095,0,"
    ],
    "CUSTOM1": ["1-0", "150-1"]
}
```

For full format details, see [`10_The NGlyph File Format.md`](./10_The%20NGlyph%20File%20Format.md).

---

## Flags/arguments

| Flag                                           | Description                                              |
| ---------------------------------------------- | -------------------------------------------------------- |
| `--update`                                     | Force re-download of `GlyphModder.py` from SebiAI’s repo |
| `--nglyph`                                     | Only make `.nglyph` files (skip audio processing and `GlyphModder.py`)        |
| `--np1`, `--np1s`, `--np2`, `--np2a`, `--np3a` | Selects which glyph config to use                        |
| *(no args)*                                    | Shows a available glyph configs and a short description of what each do.               |

---

## 💡 Example Workflow

1. Place `mysong.mp3` inside `Input/`.
2. Run:

   ```bash
   python musicViz.py --np2a
   ```
3. Wait for the FFT and processing.
4. Your results:

   ```
   Output/mysong.ogg
   Nglyph/mysong.nglyph
   ```

## Troubleshooting

* **Missing zones.config** → Create or download it from the repo.
* **GlyphModder not found** → Use `--update` to fetch it.
* **No Input files** → Add your songs in `/Input` first.



---
---
---
---
---
---

# Help me do a proper readme pls
