from PIL import Image
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
import torch


class CrackForestDataset(Dataset):
    def __init__(self, opt, phase="train"):
        super().__init__()
        self.opt = opt
        np.random.seed(opt.seed)
        self.images = np.sort(os.listdir(os.path.join(opt.dataroot, "image")))
        self.labels = np.sort(os.listdir(os.path.join(opt.dataroot, "label")))
        assert len(self.images) == len(self.labels)
        self.length = len(self.images)
        index = np.random.permutation(self.length)
        # print(index)
        self.images = [self.images[i] for i in index]
        self.labels = [self.labels[i] for i in index]

        split_num = [int(0.7 * self.length), int(0.9 * self.length)]
        if phase == "train":
            self.images = self.images[: split_num[0]]
            self.labels = self.labels[: split_num[0]]
        elif phase == "val":
            self.images = self.images[split_num[0] : split_num[1]]
            self.labels = self.labels[split_num[0] : split_num[1]]
        elif phase == "valtest":
            self.images = self.images[split_num[0] :]
            self.labels = self.labels[split_num[0] :]
        else:
            self.images = self.images[split_num[1] :]
            self.labels = self.labels[split_num[1] :]
        assert len(self.images) == len(self.labels)
        self.length = len(self.images)
        for i in range(self.length):
            assert self.images[i].split(".")[0] == self.labels[i].split(".")[0]

    def __getitem__(self, index):
        image_filepath = os.path.join(self.opt.dataroot, f"image/{self.images[index]}")
        label_filepath = os.path.join(self.opt.dataroot, f"label/{self.labels[index]}")
        image = Image.open(image_filepath).convert("RGB")
        label = Image.open(label_filepath)
        image = self.opt.transforms(image)
        label = self.opt.transforms(label)
        label[label >= 0.1] = 1.0
        label = label.squeeze()
        return image, label.long()

    def __len__(self):
        return self.length
