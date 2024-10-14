from configs.SnakeKan import Config as SnakeKanConfig
from models.SnakeKanModel import SnakeKanModel
from fvcore.nn import FlopCountAnalysis
import torch
device='cpu'
inputs = torch.randn((1, 3, 256, 256)).to(device)
opt = SnakeKanConfig()
opt.device='cpu'
model=SnakeKanModel(opt,n_channels=3,n_classes=2)
model.to(device)
flops = FlopCountAnalysis(model, inputs)
n_param = sum([p.nelement() for p in model.parameters()])
print(f'GMac:{flops.total() / (1024 * 1024 * 1024)}')
print(f'Params:{n_param}')

