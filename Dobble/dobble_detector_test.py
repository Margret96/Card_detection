import cv2
import torch
from ultralytics import YOLO  # Import YOLO from ultralytics
from torchvision import transforms
import numpy as np

# Load your YOLOv8 model
model_path = "models/test_best.pt"  # Replace with your model file
model = YOLO(model_path)  # This directly loads the YOLOv8 model

# Transformation for input to the model
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),  # Adjust based on model input size
    transforms.ToTensor(),
])

# Initialize SIFT for feature detection and matching
sift = cv2.SIFT_create()
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)  # Use KNN matching instead

# Initialize video capture
cap = cv2.VideoCapture(1)  # Use 0 for webcam

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to RGB and prepare it for the model
    input_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image_tensor = transform(input_image).unsqueeze(0)

    # Run inference
    results = model.predict(source=image_tensor, save=False, verbose=False)

    # Extract bounding boxes and labels from YOLO predictions
    boxes = results[0].boxes.xyxy.cpu().numpy()  # Get the bounding box coordinates
    scores = results[0].boxes.conf.cpu().numpy()  # Confidence scores
    labels = results[0].boxes.cls.cpu().numpy()  # Class labels

    # Filter boxes by confidence score threshold
    threshold = 0.5  # Adjust as needed
    indices = [i for i, score in enumerate(scores) if score > threshold]
    boxes = boxes[indices]

    detected_descriptors = []

    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        height, width, _ = frame.shape
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(width, x2), min(height, y2)
        symbol_roi = frame[y1:y2, x1:x2]

        if symbol_roi.size == 0:
            continue

        gray_roi = cv2.cvtColor(symbol_roi, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = sift.detectAndCompute(gray_roi, None)

        if descriptors is not None and len(descriptors) > 1:  # Ensure we have enough descriptors
            detected_descriptors.append((keypoints, descriptors, (x1, y1, x2, y2)))
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    for i in range(len(detected_descriptors)):
        for j in range(i + 1, len(detected_descriptors)):
            kp1, desc1, box1 = detected_descriptors[i]
            kp2, desc2, box2 = detected_descriptors[j]

            if desc1 is None or desc2 is None or len(desc1) < 2 or len(desc2) < 2:
                continue  # Skip if either descriptor set is empty or too small

            matches = bf.knnMatch(desc1, desc2, k=2)

            good_matches = []
            for match_pair in matches:
                if len(match_pair) == 2:  # Check that there are 2 matches in this pair
                    m, n = match_pair
                    if m.distance < 0.75 * n.distance:  # Lowe's ratio test
                        good_matches.append(m)

            if len(good_matches) > 10:
                cv2.rectangle(frame, (box1[0], box1[1]), (box1[2], box1[3]), (255, 0, 0), 2)
                cv2.rectangle(frame, (box2[0], box2[1]), (box2[2], box2[3]), (255, 0, 0), 2)

    cv2.imshow("Symbol Detection and Matching", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
