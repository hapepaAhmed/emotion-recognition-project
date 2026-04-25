import torch.nn as nn
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights


class MobileNetModel(nn.Module):
    def __init__(self, num_classes=3):
        super(MobileNetModel, self).__init__()

        # Load pretrained MobileNetV2
        self.model = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)

        # Freeze early layers (keep feature extractor stable)
        for param in self.model.features[:-2].parameters():
            param.requires_grad = False
        for param in self.model.features[-2:].parameters():
            param.requires_grad = True

        # Improved classifier head (VERY IMPORTANT for FER)
        self.model.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.model.last_channel, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.model(x)