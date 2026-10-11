"""Small CNN used by the defect detector (CPU is enough; runs on a ROCm GPU unchanged)."""
import torch
import torch.nn as nn

INPUT_SIZE = 96
MEAN = (0.5, 0.5, 0.5)
STD = (0.5, 0.5, 0.5)


def _block(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True), nn.MaxPool2d(2))


class DefectCNN(nn.Module):
    def __init__(self, n_classes: int):
        super().__init__()
        self.features = nn.Sequential(_block(3, 16), _block(16, 32), _block(32, 64), _block(64, 96))
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.2), nn.Linear(96, n_classes))

    def forward(self, x):
        return self.head(self.features(x))
