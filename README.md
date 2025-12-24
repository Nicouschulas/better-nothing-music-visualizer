# Better-nothing-music-visualizer
This script generates **NGlyph** light animations from any audio file, then runs the generated file through *SebiAi's GlyphModder* to have **better music visualization on nothing phones**. Because the one that Nothing made is *not that good* after all.

## This repo is still under construction! Feel free to contribute!
 [This is the discord thread of the project in the Nothing server.](https://discord.com/channels/930878214237200394/1434923843239280743) Feel free to jump in and join us!

# Video demo (early version of the script)
Here's a comparison between an early verizon of the script and Nothing's music visualizer. Click on the image below to open the YouTube video:

[![Watch the video](https://img.youtube.com/vi/pQZAkEl7OqQ/0.jpg)](https://www.youtube.com/watch?v=pQZAkEl7OqQ)

# Why does this exist?
For a lot of people (including me), the *Stock Glyph Music Visualization provided by Nothing* feature feels random, even if it actually isn't, it's just not obvious.

### On top of that, that feature isn't really using the whole potential of the Glyph Interface:
- Only 3 light intensities are used, like a 2 bit depth pwm,
- Looks like 25fps,
- And as I mentioned earlier, feels random most of the time.

### So, that's why I made my own music visualizer:
- It uses the full 12 bit depth pwm of the glyph interface (4096 light levels),
- It has a consistent 60 fps frame rate,
- It uses the fragmented glyphs, and the 15-zone mode of the Nothing Phone (1).
- It really syncs with the music and it's obvious, unlike Nothing's where you really need to focus to notice.

## Overview

`musicViz.py` takes an audio file (like `.mp3`, `.m4a`, `.ogg`), creates a `.nglyph` file containing the Glyph animations, and outputs a glyphed **OGG** file for playback in *Glyph composer* or in *Glyphify*.


## How it works:

* **FFT (Fast Fourier Transform)** is used to analyze frequencies in a 20ms window for each 16.666ms frame (60fps), making the visualization more accurate.
* **Frequency ranges** can be defined in `zones.config` and can be fully customized to your liking.
* The **brightness** of each glyph is defined by the peak magnitude found in the frequancy range defined in `zones.config`. It measures how loud different frequency ranges (“zones”) are.
* **Downward-only smoothing** is applied to make it look smoother while preserving reactiveness.
* A `.nglyph` file is then generated with all the brightness data (see the [NGlyph Format](#-the-nglyph-format)).
*  **SebiAi's** `GlyphModder.py` then converts the `.nglyph` file to a glyphed `.ogg` *ringtone* playable on **Nothing Phones**, containing the audio and the glyph animation.

---

# [CLICK HERE to see how to use `MusicViz.py`!](https://github.com/Aleks-Levet/better-nothing-music-visualizer/wiki/)

---
### [VirusTotal scan here](https://www.virustotal.com/gui/url/c92c1ff82b56eb60bfd1e159592d09f949f0ea2d195e01f7f5adbef0e0b0385b?nocache=1)

---
---
---
---
---

# Help me do a proper readme pls
