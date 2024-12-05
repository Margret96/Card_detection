import cv2
import numpy as np

# Constants.
INPUT_WIDTH = 640
INPUT_HEIGHT = 640
SCORE_THRESHOLD = 0.5
NMS_THRESHOLD = 0.45
CONFIDENCE_THRESHOLD = 0.45

# Text parameters.
FONT_FACE = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.7
THICKNESS = 1

# Colors
BLACK  = (0,0,0)
BLUE   = (255,178,50)
YELLOW = (0,255,255)
RED = (0,0,255)

# Load class names
classes = ['10c', '10d', '10h', '10s', '2c', '2d', '2h', '2s', '3c', '3d', '3h', '3s', 
           '4c', '4d', '4h', '4s', '5c', '5d', '5h', '5s', '6c', '6d', '6h', '6s', 
           '7c', '7d', '7h', '7s', '8c', '8d', '8h', '8s', '9c', '9d', '9h', '9s', 
           'Ac', 'Ad', 'Ah', 'As', 'Jc', 'Jd', 'Jh', 'Js', 'Kc', 'Kd', 'Kh', 'Ks', 
           'Qc', 'Qd', 'Qh', 'Qs']

def draw_label(input_image, label, left, top):
    """Draw text onto image at location."""
    
    # Get text size.
    text_size = cv2.getTextSize(label, FONT_FACE, FONT_SCALE, THICKNESS)
    dim, baseline = text_size[0], text_size[1]
    # Use text size to create a BLACK rectangle. 
    cv2.rectangle(input_image, (left, top), (left + dim[0], top + dim[1] + baseline), BLACK, cv2.FILLED)
    # Display text inside the rectangle.
    cv2.putText(input_image, label, (left, top + dim[1]), FONT_FACE, FONT_SCALE, YELLOW, THICKNESS, cv2.LINE_AA)


def pre_process(input_image, net):
    # Create a 4D blob from a frame.
    blob = cv2.dnn.blobFromImage(input_image, 1/255, (INPUT_WIDTH, INPUT_HEIGHT), [0,0,0], swapRB=True, crop=False)

    # Sets the input to the network.
    net.setInput(blob)

    # Runs the forward pass to get output of the output layers.
    output_layers = net.getUnconnectedOutLayersNames()
    print(f"Output Layers: {output_layers}")
    outputs = net.forward(output_layers)
    # print(outputs[0].shape)

    return outputs


def post_process(input_image, outputs):
    """Post-process the model's outputs."""
    class_ids = []
    confidences = []
    boxes = []

    image_height, image_width = input_image.shape[:2]
    x_factor = image_width / INPUT_WIDTH
    y_factor = image_height / INPUT_HEIGHT

    # YOLOv8 outputs a single array; parse it correctly
    predictions = outputs[0]  # Get the first output layer
    for detection in predictions:
        # YOLOv8 detections: [center_x, center_y, width, height, conf, class_scores...]
        bbox = detection[:4]  # Extract bbox
        confidence = detection[4]  # Object confidence
        class_scores = detection[5:]  # Class probabilities

        # Ensure confidence is a scalar
        if float(confidence) > CONFIDENCE_THRESHOLD:
            class_id = np.argmax(class_scores)
            score = class_scores[class_id]

            if score > SCORE_THRESHOLD:
                # Convert center x, y, width, height to left, top, width, height
                cx, cy, w, h = bbox
                left = int((cx - w / 2) * x_factor)
                top = int((cy - h / 2) * y_factor)
                width = int(w * x_factor)
                height = int(h * y_factor)

                boxes.append([left, top, width, height])
                confidences.append(float(score))
                class_ids.append(class_id)

    # Perform Non-Maximum Suppression (NMS)
    indices = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
    for i in indices:
        idx = i[0]
        box = boxes[idx]
        left, top, width, height = box
        cv2.rectangle(input_image, (left, top), (left + width, top + height), BLUE, 3 * THICKNESS)
        label = f"{classes[class_ids[idx]]}:{confidences[idx]:.2f}"
        draw_label(input_image, label, left, top)

    return input_image



if __name__ == '__main__':
    model_path = "models/best.onnx"
    net = cv2.dnn.readNet(model_path)

    # Open the camera
    cap = cv2.VideoCapture(0)

    fps_timer_start = cv2.getTickCount()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        fps_timer_start = cv2.getTickCount()

        # Process image.
        detections = pre_process(frame, net)
        img = post_process(frame.copy(), detections)

        fps_timer_end = cv2.getTickCount()
        fps_total = cv2.getTickFrequency() / (fps_timer_end - fps_timer_start) # For the total time required to process a frame

        #fps = 1000 / (t * 1000.0 / cv2.getTickFrequency()) # convert ms to FPS (based on inference time, for evaluationg model inference speed)

        """
        label = 'Inference time: %.2f ms' % (t * 1000.0 / cv2.getTickFrequency())
        label_fps_inf = f"FPS (Inference): {fps:.2f}"
        label_fps_full = f"FPS (full pipeline): {fps_total:.2f}"
        
        cv2.putText(img, label, (20, 40), FONT_FACE, FONT_SCALE, RED, THICKNESS, cv2.LINE_AA)
        cv2.putText(img, label_fps_inf, (20, 70), FONT_FACE, FONT_SCALE, RED, THICKNESS, cv2.LINE_AA)
        cv2.putText(img, label_fps_full, (20, 100), FONT_FACE, FONT_SCALE, RED, THICKNESS, cv2.LINE_AA)

        print(label)
        print(label_fps_inf)
        print(label_fps_full)
        """

        cv2.imshow('Output', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()    
    cv2.destroyAllWindows()