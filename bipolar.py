import torch
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.datasets import VOCDetection
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

NUM_CLASSES = 21
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_transforms():
    return transforms.Compose([transforms.ToTensor()])

print("Preparing dataset...")
voc_dataset = VOCDetection(
    root='data',
    year='2012',
    image_set='train',
    download=True,
    transform=load_transforms()
)

voc_loader = DataLoader(voc_dataset, batch_size=2, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))

def build_detection_model():
    cnn_backbone = resnet_fpn_backbone('resnet50', pretrained=True)
    detection_model = FasterRCNN(cnn_backbone, num_classes=NUM_CLASSES)
    return detection_model

detector = build_detection_model().to(DEVICE)

trainable_params = [param for param in detector.parameters() if param.requires_grad]
optimizer = torch.optim.SGD(trainable_params, lr=0.005, momentum=0.9, weight_decay=0.0005)

print("Commencing training phase...")
epochs = 2
detector.train()
for epoch in range(epochs):
    for batch_images, _ in voc_loader:
        inputs = [img.to(DEVICE) for img in batch_images]
        dummy_targets = [{
            'boxes': torch.tensor([[10, 20, 100, 150]], dtype=torch.float32).to(DEVICE),
            'labels': torch.tensor([1], dtype=torch.int64).to(DEVICE)
        } for _ in inputs]

        loss_outputs = detector(inputs, dummy_targets)
        total_loss = sum(val for val in loss_outputs.values())

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

    print(f"Epoch {epoch + 1}/{epochs}, Total Loss: {total_loss.item():.4f}")

torch.save(detector.state_dict(), "fasterrcnn_resnet50_fpn.pth")
print("Trained model saved successfully.")

def run_inference(model, image):
    model.eval()
    with torch.no_grad():
        output = model([image.to(DEVICE)])
    return output

def visualize_output(image_tensor, prediction):
    img_array = image_tensor.mul(255).permute(1, 2, 0).byte().cpu().numpy()
    plt.imshow(img_array)
    for bbox in prediction[0]['boxes']:
        x_min, y_min, x_max, y_max = bbox.int().cpu().numpy()
        plt.gca().add_patch(plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                                          fill=False, edgecolor='red', linewidth=2))
    plt.axis('off')
    plt.show()

image_sample, _ = voc_dataset[0]
pred_result = run_inference(detector, image_sample)
visualize_output(image_sample, pred_result)
