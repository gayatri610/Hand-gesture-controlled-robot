This project implements a hand gesture recognition system using OpenCV and MediaPipe to detect and interpret hand gestures for directional control.
The program uses the webcam feed to identify specific gestures that can trigger directional commands such as "Move Forward," "Move Right," "Move Left," "Move Backward," and "Stop."

## Features
- Detects hand gestures and interprets directions (forward, backward, left, right, stop).
- Uses MediaPipe's hand landmarks to calculate angles, directions, and palm area.
- Provides a simple, real-time output of recognized gestures.
## Dependencies
- Python 3.8+
- OpenCV
- MediaPipe
- NumPy
**Installations**
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/hand-gesture-recognizer.git
   cd hand-gesture-recognizer
   ```
2. Install the required dependencies:
   ```bash
   pip install opencv-python mediapipe numpy
   ```
**Usage**
1. Run the script:
   ```bash
   python zuzu_opencv.py
   ```
2. The application will open your webcam and display detected hand gestures on the screen. 
   - Each detected direction will be shown in red text on the display.
3. Use the following gestures for each command:
   - Move Forward: All fingers pointing upward
   - Move Right: Four horizontal fingers
   - Move Left: Angle indicating left with downward fingers
   - Move Backward: Five downward fingers
   - Stop: Four downward fingers and one upward finger

Files
- `zuzu_opencv.py`: Main script for gesture detection.
Note
- This program relies on the accurate detection of hand landmarks, which may vary based on lighting and camera quality.
- Ensure the camera is positioned to clearly capture your hand gestures for the best performance.

