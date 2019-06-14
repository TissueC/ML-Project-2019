"""
To calculate the accuracy before defense and after defense.
"""

import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import utils
from vgg_19_net import *
from constants import *
from dataset import *
from tqdm import tqdm

batch_size = 128

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

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.layer1 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 3, 2, 1, 1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.ConvTranspose2d(128, 128, 3, 1, 1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.ConvTranspose2d(128, 64, 3, 1, 1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(64, 64, 3, 1, 1),
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
        #bs = x.shape[0]
        out = x.view(-1, 256, 11, 11)
        out = self.layer1(out)
        out = self.layer2(out)
        return out



model = new_vgg()
#model = VGG("VGG19")
model.load_state_dict(torch.load("./models/addggggg68.pth"))
if use_cuda:
    model.cuda()
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
criterion = nn.CrossEntropyLoss()

dsetpath = [r"D:\test_dataset\raw",r"D:\test_dataset\attack",r"D:\test_dataset\label.npy"]
datas = targetpariDataset(dsetpath[0], dsetpath[1], dsetpath[2])
testloader = DataLoader(dataset=datas,batch_size=128,shuffle=True)

def validate_basic():
    model.eval()
    test_loss = 0
    correct2 = 0
    correct1 = 0
    prs=[[0 for i in range(7)] for j in range(7)]
    for bidx,(data1,data2,target) in tqdm(enumerate(testloader)):

        if use_cuda:
            data1, data2, target = data1.cuda(), data2.cuda(), target.cuda()
        with torch.no_grad():
            data1, data2, target =Variable(data1),Variable(data2),Variable(target)
            output = model(data2)
            test_loss += criterion(output, target)
            pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct2 += pred.eq(target.data.view_as(pred)).cpu().sum()
            for i, j in zip(pred,target):
                prs[i][j] += 1

            output = model(data1)
            test_loss += criterion(output, target)
            pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct1 += pred.eq(target.data.view_as(pred)).cpu().sum()

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct1, len(testloader.dataset),
        100. * correct1 / len(testloader.dataset)))
    acc = float(1.0 * correct1) / len(testloader.dataset)
    print(acc)
    print("---------------------------------------------------------------")
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct2, len(testloader.dataset),
        100. * correct2 / len(testloader.dataset)))
    acc = float(1.0 * correct2) / len(testloader.dataset)
    print(acc)
    return prs

def validate_newvgg():
    model.eval()
    test_loss = 0
    correct2 = 0
    correct1 = 0
    for bidx,(data1,data2,target) in tqdm(enumerate(testloader)):

        if use_cuda:
            data1, data2, target = data1.cuda(), data2.cuda(), target.cuda()
        with torch.no_grad():
            data1, data2, target =Variable(data1),Variable(data2),Variable(target)
            _,output = model(data2)
            test_loss += criterion(output, target)
            pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct2 += pred.eq(target.data.view_as(pred)).cpu().sum()

            _, output = model(data1)
            test_loss += criterion(output, target)
            pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct1 += pred.eq(target.data.view_as(pred)).cpu().sum()

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct1, len(testloader.dataset),
        100. * correct1 / len(testloader.dataset)))
    acc = float(1.0 * correct1) / len(testloader.dataset)
    print(acc)
    print("---------------------------------------------------------------")
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct2, len(testloader.dataset),
        100. * correct2 / len(testloader.dataset)))
    acc = float(1.0 * correct2) / len(testloader.dataset)
    print(acc)


def validate_dae():
    model.eval()
    test_loss = 0
    correct2 = 0
    correct1 = 0
    encoder, decoder = torch.load("./models/deno_autoencoder.pkl")
    encoder.cuda()
    decoder.cuda()
    for bidx, (data1, data2, target) in tqdm(enumerate(testloader)):
        if use_cuda:
            data1, data2, target =data1.cuda(), data2.cuda(), target.cuda()
        with torch.no_grad():
            noise = torch.rand(len(data2), 3, 44, 44).cuda()
            data2 = torch.add(0.05*noise,data2)
            data1 = torch.add(0.05*noise, data1)
            data1, data2, target = Variable(data1), Variable(data2), Variable(target)
            data1 = decoder(encoder(data1))
            data2 = decoder(encoder(data2))
            """
            from PIL import Image
            tmpim1 = 256*np.array(data1[0].cpu()).transpose(1, 2, 0)
            tmpim2 = 256*np.array(data2[0].cpu()).transpose(1, 2, 0)
            tmpim1 = Image.fromarray(tmpim1.astype(np.uint8))
            tmpim2 = Image.fromarray(tmpim2.astype(np.uint8))
            tmpim1.show()
            tmpim2.show()
            """
            output = model(data2)
            test_loss += criterion(output, target)
            pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct2 += pred.eq(target.data.view_as(pred)).cpu().sum()

            output = model(data1)
            test_loss += criterion(output, target)
            pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct1 += pred.eq(target.data.view_as(pred)).cpu().sum()

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct1, len(testloader.dataset),
        100. * correct1 / len(testloader.dataset)))
    acc = float(1.0 * correct1) / len(testloader.dataset)
    print(acc)
    print("---------------------------------------------------------------")
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct2, len(testloader.dataset),
        100. * correct2 / len(testloader.dataset)))
    acc = float(1.0 * correct2) / len(testloader.dataset)
    print(acc)

prs = validatedaedae()
print(prs)