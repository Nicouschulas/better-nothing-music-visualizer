# Better Nothing Music Visualizer

This script generates **NGlyph** light animations from any audio file, then runs the generated file through *SebiAi’s GlyphModder* to create a **better music visualization on Nothing phones**.  
Because the one made by Nothing is *not that good* after all.

---

## 🚧 This repository is still under construction! Feel free to contribute!

💬 [This is the Discord thread for the project in the Nothing server.](https://discord.com/channels/930878214237200394/1434923843239280743)  
Feel free to jump in and join us!

---

## Video demo (early version of the script)

Here’s a comparison between an early version of this script and Nothing’s stock music visualizer.  
Click the image below to watch the YouTube video:

[![Watch the video](https://img.youtube.com/vi/pQZAkEl7OqQ/0.jpg)](https://www.youtube.com/watch?v=pQZAkEl7OqQ)

---

## Why does this exist?

For a lot of people (including me), the *stock Glyph Music Visualization provided by Nothing* feels random.  
Even if it technically isn’t, the visual response to music just isn’t very obvious.

### On top of that, the feature isn’t really using the full potential of the Glyph Interface:

- Only **3 light intensity levels** are used (roughly a **2-bit PWM depth**)
- Looks like it runs at around **25 FPS**
- As mentioned earlier, it feels random most of the time

### So that’s why I made my own music visualizer:

- Uses the **full 12-bit PWM depth** of the Glyph Interface (**4096 light levels**)
- Runs at a consistent **60 FPS**
- Uses **fragmented glyphs** and the **15-zone mode** of the Nothing Phone (1)
- Clearly syncs with the music — unlike Nothing’s, where you really need to focus to notice it

---

## Overview

`musicViz.py` takes an audio file (such as `.mp3`, `.m4a`, or `.ogg`), generates a `.nglyph` file containing the Glyph animations, and outputs a **glyphed OGG** file for playback in:

- *Glyph Composer*
- *Glyphify*
- Or directly on Nothing phones

---

## How it works

- **FFT (Fast Fourier Transform)** is used to analyze frequencies in a **20 ms window** for each **16.666 ms frame** (60 FPS), making the visualization more accurate
- **Frequency ranges** can be defined in `zones.config` and are fully customizable
- The **brightness** of each glyph is defined by the **peak magnitude** found in its assigned frequency range  
  This measures how loud different frequency “zones” are
- **Downward-only smoothing** is applied to make the animation smoother while preserving responsiveness
- A `.nglyph` file is generated containing all brightness data  
  (see the [NGlyph Format](#-the-nglyph-format))
- **SebiAi’s** `GlyphModder.py` converts the `.nglyph` file into a **glyphed `.ogg` ringtone** playable on **Nothing Phones**, containing both:
  - The audio
  - The synchronized Glyph animation

---

## 📖 Usage

👉 **[CLICK HERE to see how to use `musicViz.py`](https://github.com/Aleks-Levet/better-nothing-music-visualizer/wiki/)**

---

## 🔒 Security

🛡️ **VirusTotal scan:**  
https://www.virustotal.com/gui/url/c92c1ff82b56eb60bfd1e159592d09f949f0ea2d195e01f7f5adbef0e0b0385b?nocache=1

---

## Contributing

Contributions are very welcome!

- Open issues
- Submit pull requests
- Suggest improvements
- Experiment with new visualization ideas

---

**Made for Nothing phones, but actually using the Glyph Interface properly.**

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
