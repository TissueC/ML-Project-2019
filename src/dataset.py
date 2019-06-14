import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset,DataLoader
import pandas as pd
import numpy as np
from PIL import Image
import cv2
import os
from torchvision import transforms

class ferDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        :param root_dir: path to the csv file with annotated data
        :param transform:
        """
        self.alld = pd.read_csv(root_dir)
        self.pixels = self.alld['pixels'].values
        self.emotions = self.alld['emotion'].values
        self.root_d = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.alld)

    def __getitem__(self, idx):
        tmpdata = self.pixels[idx]
        tmpdata = str.split(tmpdata)
        for i in range(len(tmpdata)):
            tmpdata[i] = int(tmpdata[i])
        tmplabel = self.emotions[idx]
        tmpdata = np.array(tmpdata)
        tmpdata = tmpdata.reshape((48,48))
        tmpdata = tmpdata[:,:,np.newaxis]
        tmpdata = np.concatenate((tmpdata,tmpdata,tmpdata), axis=2)
        tmpdata = tmpdata.astype(np.uint8)
        tmpdata = Image.fromarray(tmpdata)
        if self.transform:
            tmpdata = self.transform(tmpdata)
        #print(tmpdata)
        #print(tmplabel)
        return tmpdata,tmplabel
class pariDataset(Dataset):
    def __init__(self, root_dir, target_dir):
        self.dsets = os.listdir(root_dir)
        self.dset1 = [os.path.join(root_dir,i) for i in self.dsets]
        self.dset2 = [os.path.join(target_dir,i) for i in self.dsets]

    def __len__(self):
        return len(self.dsets)

    def __getitem__(self, idx):
        img1 = Image.open(self.dset1[idx])
        img2 = Image.open(self.dset2[idx])
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            # normTransform
        ])
        img1 = np.array(img1)
        img2 = np.array(img2)
        #print(img1.shape)
        img1 = transform_train(img1)
        img2 = transform_train(img2)
        return img1, img2

class targetpariDataset(Dataset):
    def __init__(self, root_dir, target_dir, target):
        self.dsets = os.listdir(root_dir)
        self.dset1 = [os.path.join(root_dir,i) for i in self.dsets]
        self.dset2 = [os.path.join(target_dir,i) for i in self.dsets]
        tmptarget = np.load(target)
        self.target = [tmptarget[int(i[:-4])] for i in self.dsets]
        print(self.target)
    def __len__(self):
        return len(self.dsets)

    def __getitem__(self, idx):
        img1 = Image.open(self.dset1[idx])
        img2 = Image.open(self.dset2[idx])
        tar = self.target[idx].astype(np.int64)
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            # normTransform
        ])
        img1 = np.array(img1)
        img2 = np.array(img2)
        #print(img1.shape)
        img1 = transform_train(img1)
        #print(img1.shape)
        img2 = transform_train(img2)
        return img1, img2, tar


if __name__ =="__main__":
    datas = ferDataset(root_dir="./datas/valdata.csv")
    for i,d in enumerate(datas):
        x, y = d
        print(y)