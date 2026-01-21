import cv2
import torch
import numpy as np
import tensorflow as tf
import pyttsx3
import threading
import time
from ultralytics import YOLO

# Load models
model = YOLO("/Users/harinisri/Documents/docs/final year project/final_code/yolo11nbest.pt")
dismodel = tf.keras.models.load_model("/Users/harinisri/Documents/docs/final year project/final_code/depth_estimation_model.h5", compile=False)

# Initialize text-to-speech
engine = pyttsx3.init()
engine.setProperty('rate', 150)
engine.setProperty('volume', 1.0)

def classify_distance(bbox_height, frame_height):
    ratio = bbox_height / frame_height  # Bounding box height as fraction of frame height

    if ratio > 0.45:
        return "Very Near", (0, 0, 255)  # Red
    elif ratio > 0.30:
        return "Near", (0, 255, 255)  # Yellow
    elif ratio > 0.15:
        return "Far", (0, 255, 0)  # Green
    else:
        return "Very Far", (255, 0, 0)  # Blue

# Function to determine object position in frame
def get_position(x1, x2, frame_width):
    center_x = (x1 + x2) / 2
    if center_x < frame_width * 0.33:
        return "left"
    elif center_x > frame_width * 0.66:
        return "right"
    else:
        return "center"
    
# Function to estimate object proximity
def get_proximity(y2, y1, frame_height):
    bbox_height = y2 - y1
    return classify_distance(bbox_height, frame_height)[0]  # Extracts distance label

# Function to calculate impact based on object type
def calculate_impact(class_label):
    if class_label in ["car", "bus", "truck"]:
        return 10  # High-risk static object
    elif class_label in ["person", "bicycle"]:
        return 5  # Medium-risk dynamic object
    elif class_label in ["traffic cone", "dog"]:
        return 2  # Low-risk
    else:
        return 1  # Default minimal impact    

# Function to assess risk in a given direction
def assess_risk(direction, detected_objects):
    risk_score = 0
    for obj in detected_objects:
        obj_position, obj_distance, obj_impact = obj
        if direction == "left" and obj_position in ["left", "center"]:
            risk_score += weighted_risk(obj_distance, obj_impact)
        if direction == "right" and obj_position in ["right", "center"]:
            risk_score += weighted_risk(obj_distance, obj_impact)
    return risk_score

# Function to assign risk scores based on proximity
def weighted_risk(distance, impact):
    if distance == "Very Near":
        return impact * 3
    elif distance == "Near":
        return impact * 2
    elif distance == "Far":
        return impact * 1
    else:
        return 0

# Navigation logic function
def navigation_logic(detected_objects):
    risk_scores = {"left": 0, "right": 0}
    for obj_position, obj_distance, obj_impact in detected_objects:
        if obj_position == "center" and obj_distance == "Very Near":
            risk_scores["left"] += assess_risk("left", detected_objects)
            risk_scores["right"] += assess_risk("right", detected_objects)
            return "Move Left" if risk_scores["left"] < risk_scores["right"] else "Move Right"
    return "Proceed Forward"

cap = cv2.VideoCapture(0)
class_names = model.names  # Get class labels (COCO dataset)

# Shared variable to store navigation instruction
current_instruction = "Proceed Forward"
instruction_lock = threading.Lock()

# Function to speak instructions
def speak_instruction():
    global current_instruction
    while True:
        with instruction_lock:
            instruction = current_instruction  # Get the latest instruction
        
        engine.say(instruction)
        engine.runAndWait()
        time.sleep(1)  # Prevents overlapping speech

# Default instruction every 60 seconds
def default_instruction():
    global current_instruction
    while True:
        time.sleep(60)  # Wait for 60 seconds
        with instruction_lock:
            if current_instruction == "Proceed Forward":
                engine.say("Proceed Forward")
                engine.runAndWait()

# Start the threads
speech_thread = threading.Thread(target=speak_instruction, daemon=True)
default_thread = threading.Thread(target=default_instruction, daemon=True)
speech_thread.start()
default_thread.start()

# Navigation logic function
def navigation_logic(detected_objects):
    risk_scores = {"left": 0, "right": 0}
    
    for obj_position, obj_distance, obj_impact in detected_objects:
        if obj_position == "center" and obj_distance == "Very Near":
            risk_scores["left"] += assess_risk("left", detected_objects)
            risk_scores["right"] += assess_risk("right", detected_objects)
            return "Move Left" if risk_scores["left"] < risk_scores["right"] else "Move Right"
    
    return "Proceed Forward"

# Process video in real-time
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # End of video

    frame_height, frame_width, _ = frame.shape  # Get frame dimensions
    results = model(frame)  # Run YOLO detection
    detected_objects = []

    for result in results:
        detections = result.boxes.xyxy.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy()

        for i, box in enumerate(detections):
            x1, y1, x2, y2 = map(int, box[:4])
            class_id = int(class_ids[i])
            class_label = class_names[class_id]
            obj_position = get_position(x1, x2, frame_width)
            obj_distance = get_proximity(y2, y1, frame_height)
            obj_impact = calculate_impact(class_label)
            detected_objects.append((obj_position, obj_distance, obj_impact))

            # Draw bounding box and label
            _, color = classify_distance(y2 - y1, frame_height)
            display_text = f"{class_label} - {obj_distance}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, display_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    # Determine navigation direction
    direction = navigation_logic(detected_objects)
    cv2.putText(frame, f"Direction: {direction}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

    # Update the shared instruction variable
    with instruction_lock:
        current_instruction = direction
    
    # Display the processed frame
    cv2.imshow("Navigation Assistance", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
