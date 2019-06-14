"""
Denoising AutoEncoder
"""
import torch
import torch.nn as nn
import torch.utils as utils
from torch.autograd import Variable
import torchvision.datasets as dset
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from dataset import *

epoch = 70
batch_size = 128
learning_rate = 0.001

transform_train = transforms.Compose([
    transforms.RandomCrop(44),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    #normTransform
])

transform_test = transforms.Compose([
transforms.TenCrop(44),
transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
    #normTransform
])

datas = pariDataset(r"D:\test_dataset\raw",r"D:\test_dataset\attack")
dataloader = DataLoader(dataset=datas,batch_size=128,shuffle=True)

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),  # batch x 32 x 28 x 28
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, 3, padding=1),  # batch x 32 x 28 x 28
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, 3, padding=1),  # batch x 64 x 28 x 28
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, 3, padding=1),  # batch x 64 x 28 x 28
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, 2)  # batch x 64 x 14 x 14
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),  # batch x 128 x 14 x 14
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, 3, padding=1),  # batch x 128 x 14 x 14
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, padding=1),  # batch x 256 x 7 x 7
            nn.ReLU()
        )

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(batch_size, -1)
        return out

encoder = Encoder().cuda()

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.layer1 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 3, 2, 1, 1),  # batch x 128 x 14 x 14
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.ConvTranspose2d(128, 128, 3, 1, 1),  # batch x 128 x 14 x 14
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.ConvTranspose2d(128, 64, 3, 1, 1),  # batch x 64 x 14 x 14
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(64, 64, 3, 1, 1),  # batch x 64 x 14 x 14
            nn.ReLU(),
            nn.BatchNorm2d(64)
        )
        self.layer2 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3, 1, 1),  # batch x 32 x 14 x 14
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.ConvTranspose2d(32, 32, 3, 1, 1),  # batch x 32 x 14 x 14
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.ConvTranspose2d(32, 3, 3, 2, 1, 1),  # batch x 1 x 28 x 28
            nn.ReLU()
        )

    def forward(self, x):
        out = x.view(-1, 256, 11, 11)
        out = self.layer1(out)
        out = self.layer2(out)
        return out


decoder = Decoder().cuda()
parameters = list(encoder.parameters()) + list(decoder.parameters())
loss_func = nn.MSELoss()
optimizer = torch.optim.Adam(parameters, lr=learning_rate)

# train encoder and decoder

for i in range(epoch):
    print(i)
    for image,image_n in dataloader:
        #image_a, image_b
        nl = image.shape[0]
        noise = torch.rand(nl, 3, 44, 44)
        image_n = torch.add(image_n, 0.10 * noise)
        image = Variable(image).cuda()
        image_n = Variable(image_n).cuda()
        optimizer.zero_grad()
        output = encoder(image_n)
        output = decoder(output)
        loss = loss_func(output, image)
        loss.backward()
        optimizer.step()
        print(loss.data)

    torch.save([encoder, decoder], './models/deno_autoencoder.pkl')
