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

def pred_one_epoch(opt, model, dataloader, optimizer, epoch, train=True):
    model.train()
    losses = []
    mious = []
    dataloader = tqdm(dataloader)
    prefix = 'Train' if train else 'Test'
    for images,labels in dataloader:
        images,labels = images.to(opt.device),labels.to(opt.device)
        pred_masks = model(images)
        loss = opt.criterion(pred_masks, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        mious.append(calculate_miou(pred_masks, labels,opt.n_classes))
        dataloader.set_description(f"Phase: {prefix} | Epoch: {epoch} | Loss: {np.mean(losses)} | MIou: {np.mean(mious)}")
    return np.mean(mious)

if __name__ == '__main__':
    args = getArgs()

    if args.name == 'baseline':
        opt = BaselineConfig()
    else:
        opt = SnakeKanConfig()

    model = opt.model(n_channels=3,n_classes=opt.n_classes)
    model.load_state_dict(torch.load(opt.eval_weight,map_location=opt.device))
    model = model.to(opt.device)
    model.eval()
    optimizer = opt.optimizer(model.parameters(),lr=1e-4)
    testDataset = opt.dataset(opt,phase='test')
    testLoader = torch.utils.data.DataLoader(testDataset,batch_size=opt.batch_size,shuffle=False)
    miou = pred_one_epoch(opt,model,testLoader,optimizer,0,train=False)
    print(miou)