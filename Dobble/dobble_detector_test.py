import cv2
import torch
from torchvision import transforms
import numpy as np

# Load your pre-trained PyTorch model
model_path = "your_model.pt"  # Replace with your model file
model = torch.load(model_path)
model.eval()

# Transformation for input to the model
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),  # Adjust based on model input size
    transforms.ToTensor(),
])

# Initialize SIFT for feature detection and matching
sift = cv2.SIFT_create()
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

# Initialize video capture
cap = cv2.VideoCapture(0)  # Use 0 for webcam

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to RGB and prepare it for the model
    input_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image_tensor = transform(input_image).unsqueeze(0)

    # Run inference
    with torch.no_grad():
        predictions = model(image_tensor)[0]

    # Extract bounding boxes and labels
    boxes = predictions['boxes'].cpu().numpy()
    scores = predictions['scores'].cpu().numpy()
    labels = predictions['labels'].cpu().numpy()

    # Filter boxes by confidence score threshold
    threshold = 0.5  # Adjust as needed
    indices = [i for i, score in enumerate(scores) if score > threshold]
    boxes = boxes[indices]
    labels = labels[indices]

    # Initialize a list for descriptors of detected symbols
    detected_descriptors = []

    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Extract the symbol region and compute SIFT descriptors
        symbol_roi = frame[y1:y2, x1:x2]
        gray_roi = cv2.cvtColor(symbol_roi, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = sift.detectAndCompute(gray_roi, None)

        if descriptors is not None:
            detected_descriptors.append((keypoints, descriptors, (x1, y1, x2, y2)))

    # Match descriptors between detected symbols
    for i in range(len(detected_descriptors)):
        for j in range(i + 1, len(detected_descriptors)):
            kp1, desc1, box1 = detected_descriptors[i]
            kp2, desc2, box2 = detected_descriptors[j]

            matches = bf.match(desc1, desc2)
            matches = sorted(matches, key=lambda x: x.distance)

            # Filter matches by distance threshold
            good_matches = [m for m in matches if m.distance < 0.75 * matches[-1].distance]

            if len(good_matches) > 10:  # Adjust the match threshold
                # Highlight matching pairs
                cv2.rectangle(frame, (box1[0], box1[1]), (box1[2], box1[3]), (255, 0, 0), 2)
                cv2.rectangle(frame, (box2[0], box2[1]), (box2[2], box2[3]), (255, 0, 0), 2)

    # Display the frame
    cv2.imshow("Symbol Detection and Matching", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
