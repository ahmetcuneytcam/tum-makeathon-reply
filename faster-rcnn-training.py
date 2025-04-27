import torch
import torch.optim as optim
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.transforms import functional as F
from torchvision.datasets import CocoDetection
from torch.utils.data import DataLoader
import torchvision.transforms as T

class CocoDetectionWrapper(CocoDetection):
    def __getitem__(self, idx):
        img, target = super().__getitem__(idx)

        boxes = []
        labels = []

        for obj in target:
            # COCO bbox format is [x_min, y_min, width, height]
            xmin, ymin, width, height = obj['bbox']
            xmax = xmin + width
            ymax = ymin + height

            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(obj['category_id'])

        # Build target in the format FasterRCNN expects
        target_new = {
            'boxes': torch.tensor(boxes, dtype=torch.float32),
            'labels': torch.tensor(labels, dtype=torch.int64)
        }

        return img, target_new


# Set device
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Define transforms
def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    """ if train:
        transforms.append(T.RandomHorizontalFlip(0.5)) """
    return T.Compose(transforms)

# Load COCO-style dataset
train_dataset = CocoDetectionWrapper(
    root='./labeled_barcode_images/makeathon-reply.v1i.coco/train',
    annFile='./labeled_barcode_images/makeathon-reply.v1i.coco/train/_annotations.coco.json',
    transform=get_transform(train=True)
)

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))

# Load pre-trained model
model = fasterrcnn_resnet50_fpn_v2(weights="DEFAULT")

# Replace the box predictor
num_classes = 2  # e.g., 2 for background + 1 object type
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)

# Move to device
model.to(device)

#####################################################################################################
############################ Optimization (Training) Section ########################################
#####################################################################################################

# Define optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = optim.SGD(params, lr=0.002, momentum=0.9, weight_decay=0.0005)

# Learning rate scheduler
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

# Training loop
num_epochs = 20

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    for images, targets in train_loader:
        images = list(img.to(device) for img in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        epoch_loss += losses.item()

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
    
    print(f"Epoch #{epoch} loss: {epoch_loss}")

    lr_scheduler.step()


torch.save(model.state_dict(), "fine_tuned_faster-rcnn.pth")