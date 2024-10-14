import torch
import torch.nn as nn
from .unet.SnakeKan import SnakeKan
class SnakeKanModel(nn.Module):
    def __init__(self,opt,n_channels=3,n_classes=1):
        super().__init__()
        self.backbone = SnakeKan(opt,n_channels=n_channels,n_classes=n_classes)

    def forward(self, x):
        return self.backbone(x)


