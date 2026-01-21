# Real-Time Navigation Assistance System for Visually Impaired Individuals


This project is a real-time navigation assistance system designed to enhance mobility and safety for visually impaired individuals. It integrates object detection, proximity estimation, and heuristic-based navigation logic to analyze the user's surroundings and provide audio feedback.

---

## Features

-  Real-time object detection using YOLOv11
-  Monocular proximity estimation based on bounding box size
-  Heuristic-based navigation logic (move left/right/forward)
-  Offline audio feedback using `pyttsx3`
-  Modular, hardware-agnostic, and adaptable for edge devices

---

## Requirements

- Python 3.8+
- OpenCV
- NumPy
- PyTorch
- `pyttsx3`
- YOLOv11 pre-trained weights

To install the dependencies, run:

```bash pip install -r requirements.txt

├── model/
│   └── yolo11nbest.pt         # Fine-tuned YOLOv11 model
├── utils/
│   ├── navigation_logic.py        # Navigation rule logic
│   ├── proximity_estimation.py    # Proximity estimation functions
│   └── tts_module.py              # Offline text-to-speech module
├── main.py                        # Main script to run the system
├── test_videos/                   # Folder for test video inputs
├── requirements.txt               # Required Python packages
└── README.md                      # This file

## For Live output
python main.py --source webcam

## For pre-recorded video
python main.py --source test_videos/sample.mp4

## Additional Arguments
python main.py --source webcam --weights model/yolov11_weights.pt --proximity --tts

## Output
Bounding boxes with labels: Very Near, Near, Far, Very Far
Real-time navigation suggestions: "Move Left", "Move Right", etc.
Audio feedback via text-to-speech

## Notes
The system supports full offline execution.
Make sure your camera is well-aligned for accurate obstacle detection.
For embedded or low-resource devices, optimize model loading and inference in main.py.