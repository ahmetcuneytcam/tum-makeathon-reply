import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.transforms import functional as F
from PIL import Image
import os
import cv2
import numpy as np
from tqdm import tqdm

# Check if GPU is available
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Load pre-trained model and move to device
model = fasterrcnn_resnet50_fpn_v2(pretrained=True)
num_classes = 2  # 1 class + background
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
model.to(device)
model.load_state_dict(torch.load("fine_tuned_faster-rcnn.pth"))
model.eval()

# Define input folder
image_folder = "./some_test_images"
output_video_path = "output_video.avi"

# Get all image file names
image_files = sorted([f for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

# Load first image to get size
first_image = cv2.imread(os.path.join(image_folder, image_files[0]))
height, width, layers = first_image.shape

fourcc = cv2.VideoWriter_fourcc(*'XVID')
video = cv2.VideoWriter(output_video_path, fourcc, 10, (width, height))  # 10 FPS

# Inference and drawing
for image_file in tqdm(image_files, desc="Processing images"):
    image_path = os.path.join(image_folder, image_file)

    # Load image with OpenCV
    image_bgr = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # Convert to tensor
    img_tensor = F.to_tensor(Image.fromarray(image_rgb)).to(device)

    with torch.no_grad():
        prediction = model([img_tensor])[0]

    prediction = {k: v.cpu() for k, v in prediction.items()}

    for box, label, score in zip(prediction['boxes'], prediction['labels'], prediction['scores']):
        if score > 0.5:  # Confidence threshold
            xmin, ymin, xmax, ymax = map(int, box)

            # Draw rectangle
            cv2.rectangle(image_bgr, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)

            # Draw label
            label_text = f'{label.item()} {score:.2f}'
            cv2.putText(image_bgr, label_text, (xmin, ymin - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    # Write frame
    video.write(image_bgr)

video.release()
print(f"Saved video to {output_video_path}!")
