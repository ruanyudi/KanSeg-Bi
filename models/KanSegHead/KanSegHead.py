import torch
from torch import nn
from .KAN import KAN
from .DSConv import DSConv_pro
#TODO

class KanSegHead(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        pass
    
    def forward(mask_embed,hs_map):
        pass