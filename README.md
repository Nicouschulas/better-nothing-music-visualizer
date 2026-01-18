1# <img src="https://raw.githubusercontent.com/Tarikul-Islam-Anik/Animated-Fluent-Emojis/master/Emojis/Travel%20and%20places/Fire.png" alt="Fire" width="35" height="35" />Better Nothing Music Visualizer
## <img src="https://raw.githubusercontent.com/Tarikul-Islam-Anik/Animated-Fluent-Emojis/master/Emojis/Smilies/Thinking%20Face.png" alt="Thinking Face" width="25" height="25" /> Why does this exist?
For a lot of people (including me), the *stock Glyph Music Visualiastion provided by Nothing* feels random.  
Even if it technically isn’t, the visual response to music just isn’t very obvious. On top of that, the feature isn’t really using the full potential of the Glyph Interface. So that’s why I made my own music visualizer.

## <img src="https://fonts.gstatic.com/s/e/notoemoji/latest/2696_fe0f/512.gif" alt="⚖" width="32" height="32"> Stock vs Better Music Visualizer
| Feature | Nothing Stock | **Better Music Visualizer** |
| :--- | :--- | :--- |
| **Light levels** | ~2-bit depth (3 light levels) | **12-bit depth (4096 light levels)** |
| **Frame Rate** | ~25 FPS | **60 FPS** |
| **Precision** | Feels random, it's hard to acually see how it's synced | **Uses FFT analysis to precisely determine the intensity of each light** |
| **Zones** | Standard, full physical glyphs are used | **Each glyph segment and sub-zone is used and controlled independently** |
| **Visualisation method** | Real-time only | **Realtime with 20ms latency, or pre-processed audio files** |

## <img src="https://fonts.gstatic.com/s/e/notoemoji/latest/1f3ac/512.gif" alt="🎬" width="35" height=""> Video demo (early version of the script)

See the difference in action! Here’s a comparison between an **early version** of this script and *Nothing’s stock music visualizer*.
Click below to watch the YouTube video:

[![Watch the video](https://img.youtube.com/vi/pQZAkEl7OqQ/0.jpg)](https://www.youtube.com/watch?v=pQZAkEl7OqQ)

## <img src="https://fonts.gstatic.com/s/e/notoemoji/latest/2049_fe0f/512.gif" alt="⁉" width="32" height="32"> What does this do ?
`musicViz.py` takes an audio file (such as `.mp3`, `.m4a`, or `.ogg`), generates a `.nglyph` file containing the Glyph animations, then runs the generated file through [*SebiAi’s GlyphModder*](https://github.com/SebiAi/custom-nothing-glyph-tools/) to create a **better music visualisation on Nothing phones**!
It then outputs a **glyphed OGG** file for playback in *Glyph Composer*, *Glyphify* or other glyph ringtone players. (A proper Nothing glyph music player app is in the works by the way!)

### <img src="https://fonts.gstatic.com/s/e/notoemoji/latest/2699_fe0f/512.gif" alt="⚙" width="25" height="25"> How it works (technically)
- **FFT (Fast Fourier Transform)** is used to analyze frequencies in a **20 ms window** for each **16.666 ms frame** (60 FPS), making the visualization more accurate
- **Frequency ranges** can be defined in `zones.config` and are fully customizable
- The **brightness** of each glyph is defined by the **peak magnitude** found in its assigned frequency range  
  This measures how loud different frequency “zones” are
- **Downward-only smoothing** is applied to make the animation smoother while preserving responsiveness
- A `.nglyph` file is generated containing all brightness data  
  (see the [NGlyph Format](https://github.com/SebiAi/custom-nothing-glyph-tools/blob/main/docs/10_The%20NGlyph%20File%20Format.md))
- **SebiAi’s** `GlyphModder.py` converts the `.nglyph` file into a **glyphed `.ogg` ringtone** playable on **Nothing Phones**, containing both:
  - The audio
  - The synchronized Glyph animation

## 📖 How to use?
The usage is pretty simple and straightforward. Nevertheless, we made a detailed wiki page which explains the installation, usage, configuration files in detail and a troubleshooting section. You can also find out how to make new presets. [Just click here to see how to use **musicViz.py** as a python script](https://github.com/Aleks-Levet/better-nothing-music-visualizer/wiki/). You know what's cool? You can convert an unlimited number of files in bulk without any trouble!

### If you want to just try the visualisation and you don't know coding / you're lazy:
We have a **discord bot** that can easily run the script on any audio file for you! It's one file at a time only, but it's very simple to use and you just need your phone!
 * Go in the [***Custom Nothing Glyph Tools* Discord server**](https://discord.gg/EmcnHqDxZt) 
 * Use the `/glyphs list` command to see available presets and their description
 * Use the `/glyphs compose` command with your audio file and your preset name
 * Wait a couple of seconds
 * And voilà! Just download the file and play it in *Glyph Composer* or *Glyphify*!

### <img src="https://raw.githubusercontent.com/Tarikul-Islam-Anik/Animated-Fluent-Emojis/master/Emojis/Objects/Musical%20Notes.png" alt="Musical Notes" width="30" height="30" /> Our Music App (coming soon):
We are currently working on a **Music Player App** that can easily run the script on any audio file for you! The best part? It's very simple to use, you just need your phone and don't need to process it manually!

**Planned features of the app:**
- Nothing aesthetic
- simple and fast
- Switch between multiple music visualisation presets with 2 taps
- fully offline (on device processing)
- Intuitive

## 📲 Supported Nothing Phone Models
Currently these models are supported:
- Nothing phone (1)
- Nothing phone (2a)
- Nothing phone (2a plus)
- Nothing phone (3)
- Nothing phone (3a)
- Nothing phone (3a pro)

**Coming soon:**
- *Nothing phone (2)*

## <img src="https://raw.githubusercontent.com/Tarikul-Islam-Anik/Animated-Fluent-Emojis/master/Emojis/Hand%20gestures/Handshake.png" alt="Handshake" width="25" height="25" /> Join our community
You want to talk or discuss? Bugs, feature requests? [Feel free to jump in and join us in the official discord thread in the Nothing server!](https://discord.com/channels/930878214237200394/1434923843239280743)

##  <img src="https://fonts.gstatic.com/s/e/notoemoji/latest/1f512/512.gif" alt="🔒" width="25" height="25"> Security
**The link to the VirusTotal scan can be found here:**  
https://www.virustotal.com/gui/url/c92c1ff82b56eb60bfd1e159592d09f949f0ea2d195e01f7f5adbef0e0b0385b?nocache=1

## 🏗️ Contributing
Come and help us! Contributions are very welcome!
You can:
- Open issues
- Submit pull requests
- Suggest improvements
- Experiment with new visualization ideas
- Create new presets
- Disscuss with the developpers

### Credits:
Here are the people involved in this project:
- [Aleks-Levet](https://github.com/Aleks-Levet) (founder and coordinator, main idea, owner)
- [Nicouschulas](https://github.com/Nicouschulas) (Readme & Wiki enhancements)
- [Interlastic](https://github.com/Interlastic) (Discord Bot to try the script easily)
- [SebiAi](https://github.com/SebiAi) (Glyphmodder and glyph related help)
- [Earnedel-lab](https://github.com/Earendel-lab) (Readme enhancements)
- [Blobby](https://github.com/NightBlobby) (Dev from Voidware labs making the Music app)
- [あけ なるかみ](https://github.com/Luke20YT) (Another dev from Voidware labs making the Music app)

### <img src="https://raw.githubusercontent.com/Tarikul-Islam-Anik/Animated-Fluent-Emojis/master/Emojis/Travel%20and%20places/Star.png" alt="Star" width="25" height="25" /> Star History
![Star History](https://api.star-history.com/svg?repos=Aleks-Levet/better-nothing-music-visualizer&type=Date)
