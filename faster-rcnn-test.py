import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A',
    'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# Check if GPU is available
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Load pre-trained model and move to device
model = fasterrcnn_resnet50_fpn(pretrained=True)
model.to(device)
model.eval()

# Load and preprocess an image
image_path = "./images_with_barcodes/DJI_20250424192953_0006_V.jpeg"
image = Image.open(image_path).convert("RGB")
img_tensor = F.to_tensor(image).to(device)

# Inference
with torch.no_grad():
    prediction = model([img_tensor])[0]

# Move prediction back to CPU for plotting
prediction = {k: v.cpu() for k, v in prediction.items()}

# Visualize
fig, ax = plt.subplots(1, figsize=(12, 8))
ax.imshow(image)

for box, label, score in zip(prediction['boxes'], prediction['labels'], prediction['scores']):
    if score > 0.5:  # Confidence threshold
        xmin, ymin, xmax, ymax = box
        rect = patches.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, linewidth=2,
                                 edgecolor='red', facecolor='none')
        ax.add_patch(rect)
        class_name = COCO_INSTANCE_CATEGORY_NAMES[label.item()]
        ax.text(xmin, ymin, f'{class_name} {score:.2f}', color='white',
                fontsize=12, backgroundcolor='red')

plt.axis('off')

plt.savefig("prediction.png")
print("Saved prediction to prediction.png!")
