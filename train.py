import argparse
from configs.Baseline import Config as BaselineConfig
from configs.SnakeKan import Config as SnakeKanConfig
import torch
from tqdm import tqdm
import numpy as np
from utils import calculate_miou
def getArgs():
    parser = argparse.ArgumentParser(description="命令行参数示例")
    parser.add_argument('--name', type=str, default='baseline', help='Name of model')
    args = parser.parse_args()
    return args

def pred_one_epoch(opt,model, train_loader, optimizer, epoch,train=True):
    model.train()
    losses = []
    mious = []
    train_loader = tqdm(train_loader)
    prefix = 'Train' if train else 'Test'
    for images,labels in train_loader:
        images,labels = images.to(opt.device),labels.to(opt.device)
        pred_masks = model(images)
        loss = opt.criterion(pred_masks, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        mious.append(calculate_miou(pred_masks, labels,opt.n_classes))
        train_loader.set_description(f"Phase: {prefix} | Epoch: {epoch} | Loss: {np.mean(losses)} | MIou: {np.mean(mious)}")
    return np.mean(mious)

if __name__ == '__main__':
    args = getArgs()

    if args.name == 'baseline':
        opt = BaselineConfig()
    else:
        opt = SnakeKanConfig()

    model = opt.model(n_channels=3,n_classes=opt.n_classes)
    model=model.to(opt.device)
    optimizer = opt.optimizer(model.parameters(),lr=1e-4)
    trainDataset = opt.dataset(opt,phase='train')
    valDataset = opt.dataset(opt,phase='val')
    testDataset = opt.dataset(opt,phase='test')
    trainLoader = torch.utils.data.DataLoader(trainDataset, batch_size=opt.batch_size, shuffle=True)
    valLoader = torch.utils.data.DataLoader(valDataset,batch_size=opt.batch_size,shuffle=False)
    for epoch in range(opt.epochs):
        pred_one_epoch(opt,model, trainLoader, optimizer, epoch,train=True)
        torch.save(model.state_dict(), 'latest.pth')
        if (epoch+1)%5==0:
            miou = pred_one_epoch(opt,model, valLoader, optimizer, epoch,train=False)
            print(miou)