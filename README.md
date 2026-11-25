# WaveTune 🎵
### Gesture-Controlled Music Player

Real-time hand gesture recognition system to control Spotify/media playback using deep learning.

## Gestures
| Gesture | Action |
|---------|--------|
| A | ⏯ Play/Pause |
| B | ⏭ Next Track |
| C | ⏮ Previous Track |
| D | 🔊 Volume Up |
| E | 🔉 Volume Down |

## Tech Stack
- Python, TensorFlow, Keras, OpenCV, Tkinter, PyAutoGUI

## Setup
```bash
conda create -n sign_lang python=3.9
conda activate sign_lang
pip install -r requirements.txt
cd src
python preprocess.py
python train_new_model.py
python main_app.py
```
