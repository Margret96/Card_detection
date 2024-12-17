import cv2
import torch
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO("models/dobble_best_v4_combine.pt")
#model = YOLO("models/test_best.pt") # 57 class model modified to have only 1 label..

# Initialize webcam
cap = cv2.VideoCapture(1)  # 0 for default webcam

while True:
    # Read frame from webcam
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Perform object detection
    results = model(frame)

    # Check if `results` has detections
    if isinstance(results, list):
        for result in results:
            annotated_frame = result.plot()  # Process each detection result
            # Display the frame with detections
            cv2.imshow("YOLOv8 Object Detection", annotated_frame)
    else:
        # If results is a single object (not a list), process it directly
        annotated_frame = results.plot()
        cv2.imshow("YOLOv8 Object Detection", annotated_frame)

    # Break the loop if the user presses the 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources and close windows
cap.release()
cv2.destroyAllWindows()
