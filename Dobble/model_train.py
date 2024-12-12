!pip install opendatasets

! pip install ultralytics

import os
import random
import opendatasets as od
import cv2
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import torch
from ultralytics import YOLO
%matplotlib inline


# set working directory

HOME = os.getcwd()
print(HOME)

# download dataset with opendatasets
# This step varies, based on what dataset we are using...
from google.colab import files

uploaded = files.upload()

! unzip dobble_card_data.v2i.yolov8.zip

# define train, valid, test directories


train_images = os.path.join(HOME, "train/images")
train_labels = os.path.join(HOME, "train/labels")

valid_images = os.path.join(HOME, "valid/images")
valid_labels = os.path.join(HOME, "valid/labels")

test_images = os.path.join(HOME, "test/images")
test_labels = os.path.join(HOME, "test/labels")

yaml_path = os.path.join(HOME, "data.yaml")

print(train_images)
print(valid_images)
print(test_images)

yaml_path

# Define the YAML content as a string
yaml_content = """
train: /content/train/images
val: /content/valid/images
test: /content/test/images

nc: 57
names: ['Anchor', 'Apple', 'Baby bird', 'Baby bottle', 'Bomb', 'Cactus', 'Candle', 'Carrot', 'Cheese', 'Chess Knight', 'Clock', 'Clown', 'Daisy flower', 'Dinosaur', 'Dobble', 'Dog', 'Dolphin', 'Dragon', 'Exclamation point', 'Fire', 'Four leaf clover', 'Ghost', 'Green Splats', 'Hammer', 'Heart', 'Ice cube', 'Igloo', 'Key', 'Lady bug', 'Light bulb', 'Lightning', 'Lips', 'Lock', 'Maple leaf', 'Moon', 'No Entry Sign', 'Orange Scarecrow man', 'Pencil', 'Purple Cat', 'Question mark', 'Scissors', 'Shades', 'Skull', 'Snowflake', 'Snowman', 'Spider', 'Spider Web', 'Sun', 'Target', 'Taxi car', 'Tortoise', 'Treble clef', 'Tree', 'Water drip', 'Yin and Yang', 'Zebra', 'eye']
"""

# Write the YAML content to a file
with open(yaml_path, 'w') as file:
    file.write(yaml_content)

# Define the labels

classes = ['Anchor', 'Apple', 'Baby bird', 'Baby bottle', 'Bomb', 'Cactus', 'Candle', 'Carrot', 'Cheese', 'Chess Knight', 'Clock', 'Clown', 'Daisy flower', 'Dinosaur', 'Dobble', 'Dog', 'Dolphin', 'Dragon', 'Exclamation point', 'Fire', 'Four leaf clover', 'Ghost', 'Green Splats', 'Hammer', 'Heart', 'Ice cube', 'Igloo', 'Key', 'Lady bug', 'Light bulb', 'Lightning', 'Lips', 'Lock', 'Maple leaf', 'Moon', 'No Entry Sign', 'Orange Scarecrow man', 'Pencil', 'Purple Cat', 'Question mark', 'Scissors', 'Shades', 'Skull', 'Snowflake', 'Snowman', 'Spider', 'Spider Web', 'Sun', 'Target', 'Taxi car', 'Tortoise', 'Treble clef', 'Tree', 'Water drip', 'Yin and Yang', 'Zebra', 'eye']

Idx2Label = {idx: label for idx, label in enumerate(classes)}
Label2Index = {label: idx for idx, label in Idx2Label.items()}

print('Index to Label Mapping:', Idx2Label)
print('Label to Index Mapping:', Label2Index)

def visualize_image_with_annotation_bboxes(image_dir, label_dir):
    # Get list of all the image files in the directory
    image_files = sorted(os.listdir(image_dir))

    # Choose 10 random image files from the list
    sample_image_files = random.sample(image_files, 12)

    # Set up the plot
    fig, axs = plt.subplots(4, 3, figsize=(15, 20))

    # Loop over the random images and plot the bounding boxes
    for i, image_file in enumerate(sample_image_files):
        row = i // 3
        col = i % 3

        # Load the image
        image_path = os.path.join(image_dir, image_file)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Load the labels for this image
        label_path = os.path.join(label_dir, image_file[:-4] + '.txt')
        f = open(label_path, 'r')

        # Loop over the labels and plot the bounding boxes
        for label in f:
            class_id, x_center, y_center, width, height = map(float, label.split())
            h, w, _ = image.shape
            x_min = int((x_center - width/2) * w)
            y_min = int((y_center - height/2) * h)
            x_max = int((x_center + width/2) * w)
            y_max = int((y_center + height/2) * h)
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv2.putText(image, Idx2Label[int(class_id)], (x_min, y_min), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)

        axs[row, col].imshow(image)
        axs[row, col].axis('off')

    plt.show()

# Read a image by path
image_path = os.path.join(train_images, os.listdir(train_images)[-1])
image = cv2.imread(image_path)

# Get the size of the image
height, width, channels = image.shape
print('The image has dimensions {}x{} and {} channels'.format(height, width, channels))

# Load a pretrained nano model
model = YOLO('yolov8n.pt')

# free up GPU memory
torch.cuda.empty_cache()

# Training the model
results = model.train(
    data= yaml_path,
    epochs = 25,
    imgsz = (height, width, channels),
    seed = 42,
    batch = 8,
    workers = 4,
    patience = 5,
    name = 'yolov8n_dobble')

# plot the result

%matplotlib inline
# read in the results.csv file as a pandas dataframe
df = pd.read_csv('/content/runs/detect/yolov8n_dobble/results.csv')
df.columns = df.columns.str.strip()

# create subplots using seaborn
fig, axs = plt.subplots(nrows=5, ncols=2, figsize=(15, 15))

# plot the columns using seaborn
sns.lineplot(x='epoch', y='train/box_loss', data=df, ax=axs[0,0])
sns.lineplot(x='epoch', y='train/cls_loss', data=df, ax=axs[0,1])
sns.lineplot(x='epoch', y='train/dfl_loss', data=df, ax=axs[1,0])
sns.lineplot(x='epoch', y='metrics/precision(B)', data=df, ax=axs[1,1])
sns.lineplot(x='epoch', y='metrics/recall(B)', data=df, ax=axs[2,0])
sns.lineplot(x='epoch', y='metrics/mAP50(B)', data=df, ax=axs[2,1])
sns.lineplot(x='epoch', y='metrics/mAP50-95(B)', data=df, ax=axs[3,0])
sns.lineplot(x='epoch', y='val/box_loss', data=df, ax=axs[3,1])
sns.lineplot(x='epoch', y='val/cls_loss', data=df, ax=axs[4,0])
sns.lineplot(x='epoch', y='val/dfl_loss', data=df, ax=axs[4,1])

# set titles and axis labels for each subplot
axs[0,0].set(title='Train Box Loss')
axs[0,1].set(title='Train Class Loss')
axs[1,0].set(title='Train DFL Loss')
axs[1,1].set(title='Metrics Precision (B)')
axs[2,0].set(title='Metrics Recall (B)')
axs[2,1].set(title='Metrics mAP50 (B)')
axs[3,0].set(title='Metrics mAP50-95 (B)')
axs[3,1].set(title='Validation Box Loss')
axs[4,0].set(title='Validation Class Loss')
axs[4,1].set(title='Validation DFL Loss')

# add suptitle and subheader
plt.suptitle('Training Metrics and Loss', fontsize=24)

# adjust top margin to make space for suptitle
plt.subplots_adjust(top=0.8)

# adjust spacing between subplots
plt.tight_layout()

plt.show()

# Loading the best performing model
model = YOLO('/content/runs/detect/yolov8n_dobble/weights/best.pt')

# Evaluating the model on test dataset
metrics = model.val(conf=0.25, split='test')

print(f"Mean Average Precision @.5:.95 : {metrics.box.map}")
print(f"Mean Average Precision @ .50   : {metrics.box.map50}")
print(f"Mean Average Precision @ .70   : {metrics.box.map75}")

# Function to perform detections with trained model
def predict_detection(image_path):
    # Read the image
    image = cv2.imread(image_path)

    # Pass the image through the detection model and get the result
    detect_result = model(image)

    # Plot the detections
    detect_image = detect_result[0].plot()

    # Convert the image to RGB format
    detect_image = cv2.cvtColor(detect_image, cv2.COLOR_BGR2RGB)

    return detect_image

# Get list of all the image files in the test directory
image_files = sorted(os.listdir(test_images))

# Choose 12 random image files from the list
sample_image_files = random.sample(image_files, 12)

# Set up the plot
fig, axs = plt.subplots(4, 3, figsize=(15, 20))

# Loop over the random images and plot the detections of the trained model
for i, image_file in enumerate(sample_image_files):
    row = i // 3
    col = i % 3

    # Load the current image and run object detection
    image_path = os.path.join(test_images, image_file)
    detect_image = predict_detection(image_path)

    axs[row, col].imshow(detect_image)
    axs[row, col].axis('off')

plt.show()

