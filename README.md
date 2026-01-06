# Better Nothing Music Visualizer
## 🤔 Why does this exist?
For a lot of people (including me), the *stock Glyph Music Visualization provided by Nothing* feels random.  
Even if it technically isn’t, the visual response to music just isn’t very obvious. On top of that, the feature isn’t really using the full potential of the Glyph Interface. So that’s why I made my own music visualizer.

## ⚖️ Stock vs Better Music Visualizer
| Feature | Nothing Stock | **Better Music Visualizer** |
| :--- | :--- | :--- |
| **Depth** | ~2-bit (3 light levels) | **12-bit (4096 light levels)** |
| **Frame Rate** | ~25 FPS | **60 FPS** |
| **Precision** | Feels random | **Syncs with FFT analysis** |
| **Zones** | Standard | **15-zone support** |

## 📺 Video demo (early version of the script)

See the difference in action! Here’s a comparison between an early version of this script and Nothing’s stock music visualizer.
Click below to watch the YouTube video:

[![Watch the video](https://img.youtube.com/vi/pQZAkEl7OqQ/0.jpg)](https://www.youtube.com/watch?v=pQZAkEl7OqQ)

## 🛠️ What it does
`musicViz.py` takes an audio file (such as `.mp3`, `.m4a`, or `.ogg`), generates a `.nglyph` file containing the Glyph animations,  then runs the generated file through *SebiAi’s GlyphModder* to create a **better music visualization on Nothing phones** 
and later outputs a **glyphed OGG** file for playback in *Glyph Composer*, *Glyphify* or directly on Nothing phones.

### ⚙️ How it works (technically)
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

## 📖 How to use
The usage is pretty simple and straightforward. Nevertheless, we made a Wiki page which explains the installation, usage, configuration files in detail and a troubleshooting section. [Just click here to see how to use **musicViz.py**](https://github.com/Aleks-Levet/better-nothing-music-visualizer/wiki/)

## 📲 Supported Models
Currently these models are supported:
- Nothing phone (1)
- Nothing phone (2a)
- Nothing phone (2a plus)
- Nothing phone (3a)
- Nothing phone (3a pro)
- Phone 2 support is coming soon!

## 🤝 Join our community
You want to talk or discuss? [Feel free to jump in and join us in the official discord thread on the nothing server!](https://discord.com/channels/930878214237200394/1434923843239280743)

## 🔒 Security
**The link to the VirusTotal scan can be found here:**  
https://www.virustotal.com/gui/url/c92c1ff82b56eb60bfd1e159592d09f949f0ea2d195e01f7f5adbef0e0b0385b?nocache=1

## 🏗️ Contributing
Contributions are very welcome! You can:
- Open issues
- Submit pull requests
- Suggest improvements
- Experiment with new visualization ideas
- Create new presets
- Disscuss with the developpers

### 📈 Star History
![Star History](https://api.star-history.com/svg?repos=Aleks-Levet/better-nothing-music-visualizer&type=Date)

## 🚧 This repository is still under construction! Feel free to contribute!
