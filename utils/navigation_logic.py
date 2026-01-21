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
            return "Move left" if risk_scores["left"] < risk_scores["right"] else "Move right"
    return "Proceed Forward"

# Process video in real-time
# Process video in real-time
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # End of video

    frame_height, frame_width, _ = frame.shape  # Get frame dimensions
    results = model(frame)  # Run YOLOv11 detection
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

    # Speak only if instruction changes and it's not 'Proceed Forward'
    if direction != last_instruction and direction != "Proceed Forward":
        engine.say(direction)
        engine.runAndWait()
        last_instruction = direction
    elif direction == "Proceed Forward":
        last_instruction = None  # Reset so that next left/right will be spoken

    # Display the processed frame
    cv2.imshow("Navigation Assistance", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break