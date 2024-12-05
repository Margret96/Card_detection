import cv2
import time
from ultralytics import YOLO


def draw_detections(frame, results):
    """
    Draw bounding boxes and labels for each detection on the frame.
    """
    for result in results:
        for box in result.boxes:
            # Extract bounding box and label information
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())  # Bounding box coordinates
            confidence = box.conf[0]  # Confidence score
            class_id = int(box.cls[0])  # Class ID
            label = f"{classes[class_id]}: {confidence:.2f}"

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # Draw label
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return frame


def calculate_fps(start_time):
    """
    Calculate and return frames per second.
    """
    end_time = time.time()
    fps = 1 / (end_time - start_time)
    return fps, end_time


def process_video(model, cap, output_video_path=None):
    """
    Process the video frame-by-frame and run YOLOv8 detection.
    """
    # Video writer setup (if saving output)
    writer = None
    if output_video_path:
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    start_time = time.time()

    while True:
        # Read frame from webcam
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Perform object detection
        results = model(frame)

        # Draw detections
        frame = draw_detections(frame, results)

        # Calculate FPS
        fps, start_time = calculate_fps(start_time)
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        # Display the frame
        cv2.imshow("YOLOv8 Object Detection", frame)

        # Save to video file if writer is enabled
        if writer:
            writer.write(frame)

        # Break loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    if writer:
        writer.release()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # Load YOLOv8 model
    model = YOLO("models/yolov8n.pt")

    # Class names (update with your own dataset classes)
    classes = [
        '10c', '10d', '10h', '10s', '2c', '2d', '2h', '2s', '3c', '3d', '3h', '3s',
        '4c', '4d', '4h', '4s', '5c', '5d', '5h', '5s', '6c', '6d', '6h', '6s',
        '7c', '7d', '7h', '7s', '8c', '8d', '8h', '8s', '9c', '9d', '9h', '9s',
        'Ac', 'Ad', 'Ah', 'As', 'Jc', 'Jd', 'Jh', 'Js', 'Kc', 'Kd', 'Kh', 'Ks',
        'Qc', 'Qd', 'Qh', 'Qs'
    ]

    # Initialize webcam
    cap = cv2.VideoCapture(0)  # Change index if using an external camera

    if not cap.isOpened():
        print("Error: Could not open camera.")
        exit()

    # Process video (optionally save to file)
    #process_video(model, cap, output_video_path="output.avi")
