import argparse
from configs.Baseline import Config as BaselineConfig
from configs.SnakeKan import Config as SnakeKanConfig
import torch
from tqdm import tqdm
import numpy as np
from utils import calculate_miou
import matplotlib.pyplot as plt
import json


train_loss = []
train_miou = []
test_loss = []
test_miou = []

def getArgs():
    parser = argparse.ArgumentParser(description="命令行参数示例")
    parser.add_argument('--name', type=str, default='snakekan', help='Name of model')
    args = parser.parse_args()
    return args

def pred_one_epoch(opt, model, dataloader, optimizer, epoch, train=True):
    losses = []
    mious = []
    dataloader = tqdm(dataloader)
    prefix = 'Train' if train else 'Test'
    for images,labels in dataloader:
        images,labels = images.to(opt.device),labels.to(opt.device)
        #plt.imshow(labels[0].cpu().numpy())
        #plt.show()
        pred_masks = model(images)
        loss = opt.criterion(pred_masks, labels)
        optimizer.zero_grad()
        loss.backward()
        if train or True:
            optimizer.step()
        losses.append(loss.item())
        mious.append(calculate_miou(pred_masks, labels,opt.n_classes))
        dataloader.set_description(f"Phase: {prefix} | Epoch: {epoch} | Loss: {np.mean(losses)} | MIou: {np.mean(mious)}")
    if train:
        train_loss.append([epoch,np.mean(losses)])
        train_miou.append([epoch,float(np.mean(mious))])
    else:
        test_loss.append([epoch,np.mean(losses)])
        test_miou.append([epoch,float(np.mean(mious))])
    return np.mean(mious)

if __name__ == '__main__':
    args = getArgs()

    if args.name == 'baseline':
        opt = BaselineConfig()
    else:
        opt = SnakeKanConfig()

    model = opt.model(opt,n_channels=3,n_classes=opt.n_classes)
    model=model.to(opt.device)
    optimizer = opt.optimizer(model.parameters(),lr=1e-4)
    trainDataset = opt.dataset(opt,phase='train')
    valDataset = opt.dataset(opt,phase='valtest')
    trainLoader = torch.utils.data.DataLoader(trainDataset, batch_size=opt.batch_size, shuffle=True)
    valLoader = torch.utils.data.DataLoader(valDataset,batch_size=1,shuffle=False)
    for epoch in range(opt.epochs):
        model.train()
        pred_one_epoch(opt,model, trainLoader, optimizer, epoch,train=True)
        torch.save(model.state_dict(), 'latest.pth')
        if (epoch+1)%2==0:
            model.eval()
            miou = pred_one_epoch(opt,model, valLoader, optimizer, epoch,train=False)
            torch.save(model.state_dict(),f'./weights/{opt.name}_{epoch}_{miou}.pth')
        json.dump(train_loss,open('logs/train_loss.json','w'))
        json.dump(train_miou,open('logs/train_miou.json','w'))
        json.dump(test_loss,open('logs/test_loss.json','w'))
        json.dump(test_miou,open('logs/test_miou.json','w'))
