import torch.nn as nn
from torchvision import models

def create_model(num_classes=7, pretrained=True):
    # Load pretrained EfficientNet-B0
    model = models.efficientnet_b0(weights='IMAGENET1K_V1' if pretrained else None)

    # Replace the final classifier layer to match our number of classes
    # EfficientNet-B0's classifier is: Sequential(Dropout, Linear(1280, 1000))
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)

    return model