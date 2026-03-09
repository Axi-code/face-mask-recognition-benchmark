import torch
from torch import nn


class Residual(nn.Module):
    def __init__(self, input_channels, num_channels, use_1conv=False, strides=1):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(
            in_channels=input_channels,
            out_channels=num_channels,
            kernel_size=3,
            padding=1,
            stride=strides,
            bias=False,
        )
        self.conv2 = nn.Conv2d(
            in_channels=num_channels,
            out_channels=num_channels,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)
        self.conv3 = (
            nn.Conv2d(
                in_channels=input_channels,
                out_channels=num_channels,
                kernel_size=1,
                stride=strides,
                bias=False,
            )
            if use_1conv
            else None
        )

    def forward(self, x):
        y = self.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        if self.conv3 is not None:
            x = self.conv3(x)
        return self.relu(y + x)


class ResNet18(nn.Module):
    def __init__(self, block=Residual, num_classes=2, classifier_dropout=0.0):
        super().__init__()
        self.b1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.b2 = nn.Sequential(
            block(64, 64, use_1conv=False, strides=1),
            block(64, 64, use_1conv=False, strides=1),
        )
        self.b3 = nn.Sequential(
            block(64, 128, use_1conv=True, strides=2),
            block(128, 128, use_1conv=False, strides=1),
        )
        self.b4 = nn.Sequential(
            block(128, 256, use_1conv=True, strides=2),
            block(256, 256, use_1conv=False, strides=1),
        )
        self.b5 = nn.Sequential(
            block(256, 512, use_1conv=True, strides=2),
            block(512, 512, use_1conv=False, strides=1),
        )
        self.b6 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(p=classifier_dropout) if classifier_dropout > 0 else nn.Identity(),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        x = self.b4(x)
        x = self.b5(x)
        return self.b6(x)


def custom_resnet18(num_classes=2, classifier_dropout=0.0):
    return ResNet18(block=Residual, num_classes=num_classes, classifier_dropout=classifier_dropout)


if __name__ == "__main__":
    model = custom_resnet18()
    dummy = torch.randn(1, 3, 224, 224)
    print(model(dummy).shape)





