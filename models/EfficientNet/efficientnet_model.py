import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights


class EfficientNetModel(nn.Module):
    def __init__(self, num_classes=3):
        super(EfficientNetModel, self).__init__()

        self.model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)

        # Freeze early layers (NOT all)
        for param in self.model.features[:-2].parameters():
            param.requires_grad = False
        for param in self.model.features[-2:].parameters():
            param.requires_grad = True
        # Improved classifier head (better for FER dataset)
        in_features = self.model.classifier[1].in_features

        self.model.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.model(x)