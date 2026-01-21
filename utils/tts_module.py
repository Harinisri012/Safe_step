import cv2
import torch
import numpy as np
from ultralytics import YOLO

import pyttsx3

# Initialize speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)
engine.setProperty('volume', 1.0)

last_instruction = None  # Tracks the last spoken direction

voices = engine.getProperty('voices')
engine.setProperty('voice', voices[132].id)  # Set to voice number 132

cap = cv2.VideoCapture(0)
class_names = model.names  # Get class labels (COCO dataset)