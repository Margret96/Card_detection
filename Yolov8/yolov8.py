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


def draw_label(input_image, label, left, top):
    """Draw text onto image at location."""
    
    # Get text size.
    text_size = cv2.getTextSize(label, FONT_FACE, FONT_SCALE, THICKNESS)
    dim, baseline = text_size[0], text_size[1]
    # Use text size to create a BLACK rectangle. 
    cv2.rectangle(input_image, (left, top), (left + dim[0], top + dim[1] + baseline), BLACK, cv2.FILLED);
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
    # Lists to hold respective values while unwrapping.
    class_ids = []
    confidences = []
    boxes = []

    # Rows.
    #rows = outputs[0].shape[1]

    image_height, image_width = input_image.shape[:2]

    # Resizing factor.
    x_factor = image_width / INPUT_WIDTH
    y_factor =  image_height / INPUT_HEIGHT

    for output in outputs:
        for detection in output[0]:
            confidence = detection[4]
            if confidence >= CONFIDENCE_THRESHOLD:
                # Extract class scores and find the best class
                class_scores = detection[5:]
                class_id = np.argmax(class_scores)
                score = class_scores[class_id]

                if score >= SCORE_THRESHOLD:
                    # Bounding box coordinates
                    cx, cy, w, h = detection[0:4]
                    left = int((cx - w / 2) * x_factor)
                    top = int((cy - h / 2) * y_factor)
                    width = int(w * x_factor)
                    height = int(h * y_factor)

                    boxes.append([left, top, width, height])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

    # Perform non maximum suppression to eliminate redundant overlapping boxes with
    # lower confidences.
    indices = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
    for i in indices:
        idx = i
        box = boxes[idx]
        left, top, width, height = box
        cv2.rectangle(input_image, (left, top), (left + width, top + height), BLUE, 3*THICKNESS)
        label = "{}:{:.2f}".format(classes[class_ids[idx]], confidences[idx])
        draw_label(input_image, label, left, top)

    return input_image


if __name__ == '__main__':
    # Load class names.
    classesFile = "coco.names"
    classes = None
    #classes = ['10c', '10d', '10h', '10s', '2c', '2d', '2h', '2s', '3c', '3d', '3h', '3s', '4c', '4d', '4h', '4s', '5c', '5d', '5h', '5s', '6c', '6d', '6h', '6s', '7c', '7d', '7h', '7s', '8c', '8d', '8h', '8s', '9c', '9d', '9h', '9s', 'Ac', 'Ad', 'Ah', 'As', 'Jc', 'Jd', 'Jh', 'Js', 'Kc', 'Kd', 'Kh', 'Ks', 'Qc', 'Qd', 'Qh', 'Qs']
    with open(classesFile, 'rt') as f:
        classes = f.read().rstrip('\n').split('\n')
    
    print(f"Number of classes loades: {len(classes)} ")

    # Open the camera
    cap = cv2.VideoCapture(0)

    fps_timer_start = cv2.getTickCount()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
    
        # Load image.
        #frame = cv2.imread('sample.jpg')

        # Give the weight files to the model and load the network using them.
        #modelWeights_slow = "models/yolov5s.onnx" # Slower model (middle of the three available)
        modelWeights_fast = "models/best.onnx" # Faster model

        #net_slow = cv2.dnn.readNet(modelWeights_small)
        net_fast = cv2.dnn.readNet(modelWeights_fast)

        fps_timer_start = cv2.getTickCount()

        # Process image.
        detections = pre_process(frame, net_fast) # Need to change according to model fast/slow
        img = post_process(frame.copy(), detections)

        # Put efficiency information. The function getPerfProfile returns the overall time for inference(t) and the timings for each of the layers(in layersTimes)
        t, _ = net_fast.getPerfProfile() # Need to change according to size of model fast/slow

        fps_timer_end = cv2.getTickCount()
        fps_total = cv2.getTickFrequency() / (fps_timer_end - fps_timer_start) # For the total time required to process a frame

        fps = 1000 / (t * 1000.0 / cv2.getTickFrequency()) # convert ms to FPS (based on inference time, for evaluationg model inference speed)

        label = 'Inference time: %.2f ms' % (t * 1000.0 / cv2.getTickFrequency())
        label_fps_inf = f"FPS (Inference): {fps:.2f}"
        label_fps_full = f"FPS (full pipeline): {fps_total:.2f}"
        
        cv2.putText(img, label, (20, 40), FONT_FACE, FONT_SCALE, RED, THICKNESS, cv2.LINE_AA)
        cv2.putText(img, label_fps_inf, (20, 70), FONT_FACE, FONT_SCALE, RED, THICKNESS, cv2.LINE_AA)
        cv2.putText(img, label_fps_full, (20, 100), FONT_FACE, FONT_SCALE, RED, THICKNESS, cv2.LINE_AA)

        print(label)
        print(label_fps_inf)
        print(label_fps_full)

        cv2.imshow('Output', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()    
    cv2.destroyAllWindows()