from dataclasses import dataclass
import torch
import torchvision
from torchvision import transforms
from models.baseline import BaseModel
from Dataset.CrackForestDataset import CrackForestDataset

@dataclass
class Config:
    name:str = 'baseline'
    dataroot:str = '/home/cavin/workspace/KanSeg-Bi/data/crackforest'
    seed:int = 42
    transforms=transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(64)
    ])
    n_classes = 2
    model:torch.nn.Module = BaseModel
    optimizer = torch.optim.Adam
    criterion = torch.nn.CrossEntropyLoss()
    dataset = CrackForestDataset
    batch_size:int = 2
    epochs:int = 10
    device:str = 'cuda' if torch.cuda.is_available() else 'cpu'
    eval_weight = './latest.pth'
