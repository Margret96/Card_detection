import cv2
import torch
from ultralytics import YOLO

# Load YOLOv8 model
#model = YOLO("models/dobble_best_small_dataset_14epoch.pt")  # 39 class label model
model = YOLO("models_okkar/dobble_best_v4_combine.pt")  # Use your trained model
#model = YOLO("models/symbol-detection-larger-dataset-25-epoch.onnx") # Model from Eyþór

# Access class names from the model
class_names = model.names  # Dictionary mapping class indices to names

# Initialize webcam
cap = cv2.VideoCapture(1)  # Adjust index for your webcam

def get_detections(results):
    """Extract bounding boxes, class labels, and confidences from YOLO results."""
    detections = []
    for result in results:
        for box in result.boxes:
            detections.append({
                "class": int(box.cls.item()),  # Class as an integer
                "bbox": box.xyxy[0].tolist(),  # Ensure bbox is converted to a list of 4 numbers
                "confidence": box.conf.item()
            })
    return detections

while True:
    # Start frame timer
    start_time = cv2.getTickCount()

    # Read frame from webcam
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Perform object detection
    results = model(frame)

    # Extract detections
    detections = get_detections(results)

    # Compare symbols for matches
    class_counts = {}
    for detection in detections:
        cls = detection['class']
        if cls in class_counts:
            class_counts[cls].append(detection)
        else:
            class_counts[cls] = [detection]

    # Find matching symbols
    matches = [cls for cls, boxes in class_counts.items() if len(boxes) > 1]
    matching_symbols = [class_names[cls] for cls in matches]  # Get the class names of matching symbols

    # Highlight matches and add labels
    for detection in detections:
        bbox = detection['bbox']
        if len(bbox) == 4:  # Ensure bbox has the correct length
            color = (0, 255, 0) if detection['class'] in matches else (255, 0, 0)
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
            
            # Add label with class name and confidence above the bounding box
            class_name = class_names[detection['class']]  # Get class name
            confidence = detection['confidence']
            label = f"{class_name} {confidence:.2f}"  # Format with confidence
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 1
            text_size = cv2.getTextSize(label, font, font_scale, thickness)[0]
            text_x = int(bbox[0])
            text_y = int(bbox[1]) - 5  # Position above the bounding box
            cv2.rectangle(frame, (text_x, text_y - text_size[1]), (text_x + text_size[0], text_y), color, -1)  # Background for text
            cv2.putText(frame, label, (text_x, text_y), font, font_scale, (255, 255, 255), thickness)

    # Calculate FPS
    end_time = cv2.getTickCount()
    time_per_frame = (end_time - start_time) / cv2.getTickFrequency()  # Time for one frame in seconds
    fps = 1.0 / time_per_frame

    # Display FPS on screen
    fps_label = f"FPS: {fps:.2f}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.75
    thickness = 2
    text_size = cv2.getTextSize(fps_label, font, font_scale, thickness)[0]
    text_x = 10  # Top-left corner
    text_y = 30  # Slightly below the top
    cv2.rectangle(frame, (text_x, text_y - text_size[1]), (text_x + text_size[0], text_y + 5), (0, 0, 0), -1)  # Background for FPS
    cv2.putText(frame, fps_label, (text_x, text_y), font, font_scale, (255, 255, 255), thickness)

    # Display matching symbols under the FPS
    if matching_symbols:
        matching_label = f"Matching: {', '.join(matching_symbols)}"
        matching_text_size = cv2.getTextSize(matching_label, font, font_scale, thickness)[0]
        matching_text_x = 10  # Same x position as FPS
        matching_text_y = text_y + 30  # Position directly below the FPS
        cv2.rectangle(frame, (matching_text_x, matching_text_y - matching_text_size[1]), 
                      (matching_text_x + matching_text_size[0], matching_text_y + 5), (0, 0, 0), -1)  # Background for matching text
        cv2.putText(frame, matching_label, (matching_text_x, matching_text_y), font, font_scale, (255, 255, 255), thickness)

    # Display the frame with detections
    cv2.imshow("YOLOv8 Object Detection", frame)

    # Break the loop if the user presses the 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources and close windows
cap.release()
cv2.destroyAllWindows()
