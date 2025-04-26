import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.transforms import functional as F
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Check if GPU is available
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Load pre-trained model and move to device
model = fasterrcnn_resnet50_fpn_v2(pretrained=True)

# Replace the box predictor
num_classes = 2  # e.g., 2 for background + 1 object type
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)

# Move to device
model.to(device)

model.load_state_dict(torch.load("fine_tuned_faster-rcnn.pth"))

model.eval()

# Load and preprocess an image
image_path = "./labeled_barcode_images/makeathon-reply.v1i.coco/train/DJI_20250424192953_0006_V_jpeg.rf.7916275dd582bb114e88af95cc4bf2af.jpg"
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
    # if score > 0.5:  # Confidence threshold
    xmin, ymin, xmax, ymax = box
    rect = patches.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, linewidth=1,
                                edgecolor='red', facecolor='none')
    ax.add_patch(rect)
    ax.text(xmin, ymin, f'{label.item()} {score:.2f}', color='white',
            fontsize=6, backgroundcolor='red')

plt.axis('off')

plt.show()

""" plt.savefig("prediction_barcodes.png", dpi=300, bbox_inches="tight")
print("Saved prediction to prediction_barcodes.png!") """