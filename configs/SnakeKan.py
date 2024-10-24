from dataclasses import dataclass
import torch
import torchvision
from torchvision import transforms
from models.SnakeKanModel import SnakeKanModel
from Dataset.CrackForestDataset import CrackForestDataset

@dataclass
class Config:
    name:str = 'SnakeKan'
    dataroot:str = '/Users/ruanyudi/PycharmProjects/KanSeg-Bi/data/crackforest'
    seed:int = 42
    transforms=transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(256)
    ])
    n_classes = 2
    model:torch.nn.Module = SnakeKanModel
    optimizer = torch.optim.Adam
    criterion = torch.nn.CrossEntropyLoss()
    dataset = CrackForestDataset
    batch_size:int = 4
    epochs:int = 150
    device:str = 'cuda' if torch.cuda.is_available() else 'cpu'
    eval_weight = './weights/SnakeKan_9_0.7625501751899719.pth'

if __name__ == '__main__':
    opt = Config()
    print(opt)
