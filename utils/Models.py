import torch
import torchvision
from torch import nn


class LungABResNet18(nn.Module):
    def __init__(self, pretrained=True, num_classes=2, dropout=0.0):
        super(LungABResNet18, self).__init__()
        self.model = torchvision.models.resnet18(pretrained=pretrained)

        # Modify the fully connected layer
        self.model.fc = nn.Linear(512, num_classes)

        if dropout > 0:
            # Add dropout to the fully connected layer (fc)
            self.model.fc = nn.Sequential(
                nn.Dropout(dropout, inplace=True),
                nn.Linear(512, num_classes)
            )

    def forward(self, x):
        return self.model(x)


class LungABResNet34(nn.Module):
    def __init__(self, pretrained=True, num_classes=2, dropout=0.0):
        super(LungABResNet34, self).__init__()
        self.model = torchvision.models.resnet34(pretrained=pretrained)

        # Modify the fully connected layer
        self.model.fc = nn.Linear(512, num_classes)

        if dropout > 0:
            # Add dropout to the fully connected layer (fc)
            self.model.fc = nn.Sequential(
                nn.Dropout(dropout, inplace=True),
                nn.Linear(512, num_classes)
            )

    def forward(self, x):
        return self.model(x)

class LungABResNet50(nn.Module):
    def __init__(self, pretrained=True, num_classes=2, dropout=0.0):
        super(LungABResNet50, self).__init__()
        self.model = torchvision.models.resnet50(pretrained=pretrained)

        # Modify the fully connected layer
        self.model.fc = nn.Linear(512, num_classes)

        if dropout > 0:
            # Add dropout to the fully connected layer (fc)
            self.model.fc = nn.Sequential(
                nn.Dropout(dropout, inplace=True),
                nn.Linear(512, num_classes)
            )

    def forward(self, x):
        return self.model(x)


class LungABInceptionV3(nn.Module):
    def __init__(self, pretrained=True, num_classes=2):
        super(LungABInceptionV3, self).__init__()
        self.model = torch.hub.load('pytorch/vision:v0.10.0', 'inception_v3', pretrained=pretrained)

        # Modify the fully connected layer
        in_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.model(x)