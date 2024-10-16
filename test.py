import argparse
from configs.Baseline import Config as BaselineConfig
from configs.SnakeKan import Config as SnakeKanConfig
import torch
from tqdm import tqdm
import numpy as np
from utils import calculate_miou
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score,recall_score,f1_score

def getArgs():
    parser = argparse.ArgumentParser(description="命令行参数示例")
    parser.add_argument('--name', type=str, default='baseline', help='Name of model')
    args = parser.parse_args()
    return args

def pred_one_epoch(opt, model, dataloader, optimizer, epoch, train=True):
    model.train()
    losses = []
    mious = []
    f1_scores = []
    precisions = []
    recalls = []
    dataloader = tqdm(dataloader)
    prefix = 'Train' if train else 'Test'
    for i,(images,labels) in enumerate(dataloader):
        images,labels = images.to(opt.device),labels.to(opt.device)
        pred_masks = model(images)
        loss = opt.criterion(pred_masks, labels)
        optimizer.zero_grad()
        loss.backward()
        if train:
            optimizer.step()
        losses.append(loss.item())
        mious.append(calculate_miou(pred_masks, labels,opt.n_classes))
        dataloader.set_description(f"Phase: {prefix} | Epoch: {epoch} | Loss: {np.mean(losses)} | MIOU: {np.mean(mious)} | {np.mean(f1_scores),np.mean(precisions),np.mean(recalls)}")
        pred_masks = torch.argmax(pred_masks,axis=1)
        
        fig,axes = plt.subplots(nrows=1,ncols=3,figsize=(15,5))
        images=images[0].cpu().permute(1,2,0).numpy()
        axes[0].imshow(images,cmap='gray')
        axes[0].set_title('input')

        labels=labels.squeeze().cpu().numpy()
        axes[1].imshow(labels,cmap='gray')
        axes[1].set_title('gt')
        
        pred_masks=pred_masks.squeeze().cpu().numpy()
        axes[2].imshow(pred_masks,cmap='gray')
        axes[2].set_title('pred')

        plt.savefig(f'./outputs/{i}.png',dpi=300)
        plt.clf()
        
        pred_masks=pred_masks.flatten()
        labels = labels.flatten()
        f1_scores.append(f1_score(pred_masks,labels))
        precisions.append(precision_score(pred_masks,labels))
        recalls.append(recall_score(pred_masks,labels))

    return np.mean(mious)

if __name__ == '__main__':
    args = getArgs()

    if args.name == 'baseline':
        opt = BaselineConfig()
    else:
        opt = SnakeKanConfig()

    opt.batch_size = 1
    model = opt.model(opt,n_channels=3,n_classes=opt.n_classes)
    model.load_state_dict(torch.load(opt.eval_weight,map_location=opt.device))
    model = model.to(opt.device)
    model.eval()
    optimizer = opt.optimizer(model.parameters(),lr=1e-4)
    testDataset = opt.dataset(opt,phase='valtest')
    testLoader = torch.utils.data.DataLoader(testDataset,batch_size=opt.batch_size,shuffle=False)
    miou = pred_one_epoch(opt,model,testLoader,optimizer,0,train=False)
    print(miou)
