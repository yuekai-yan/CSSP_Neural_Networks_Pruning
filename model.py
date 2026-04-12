import torch
import torch.nn as nn
import torch.nn.functional as F


import torch.nn as nn


class ConvBNReLU(nn.Module):
    """
    Wrapper for Conv2d + BatchNorm2d + ReLU.
    """
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=True
    ):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=bias
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.block(x)


class LinearBNReLU(nn.Module):
    """
    Wrapper for Linear + BatchNorm1d + ReLU.
    """
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(
                in_features=in_features,
                out_features=out_features,
                bias=bias
            ),
            nn.BatchNorm1d(out_features),
            nn.ReLU()
        )

    def forward(self, x):
        return self.block(x)


class VGG16(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        self.model = nn.Sequential(
            # Block 1
            ConvBNReLU(3, 64, kernel_size=3, stride=1, padding=1),
            ConvBNReLU(64, 64, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(2),

            # Block 2
            ConvBNReLU(64, 128, kernel_size=3, stride=1, padding=1),
            ConvBNReLU(128, 128, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(2),

            # Block 3
            ConvBNReLU(128, 256, kernel_size=3, stride=1, padding=1),
            ConvBNReLU(256, 256, kernel_size=3, stride=1, padding=1),
            ConvBNReLU(256, 256, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(2),

            # Block 4
            ConvBNReLU(256, 512, kernel_size=3, stride=1, padding=1),
            ConvBNReLU(512, 512, kernel_size=3, stride=1, padding=1),
            ConvBNReLU(512, 512, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(2),

            # Block 5
            ConvBNReLU(512, 512, kernel_size=3, stride=1, padding=1),
            ConvBNReLU(512, 512, kernel_size=3, stride=1, padding=1),
            ConvBNReLU(512, 512, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(2),

            # Classifier
            nn.Flatten(),          # (batch_size, 512)
            LinearBNReLU(512, 512),
            LinearBNReLU(512, 256),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.model(x)