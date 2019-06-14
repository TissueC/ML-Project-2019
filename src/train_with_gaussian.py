import torch.optim as optim
from torchvision import datasets, transforms
import utils

from vgg_19_net import *
from constants import *
from dataset import *
model = VGG("VGG19")
if use_cuda:
    model.cuda()
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
criterion = nn.CrossEntropyLoss()
#optimizer = optim.Adam(model.parameters(),lr=lr,betas=betas)

cut_size = 44
normMean = [0.49139968, 0.48215827, 0.44653124]
normStd = [0.24703233, 0.24348505, 0.26158768]
normTransform = transforms.Normalize(normMean, normStd)

transform_train = transforms.Compose([
    transforms.RandomCrop(cut_size),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    #normTransform
])

transform_test = transforms.Compose([
transforms.TenCrop(cut_size),
transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
    #normTransform
])

datas = ferDataset("./datas/training.csv",transform=transform_train)
dataloader = DataLoader(dataset=datas,batch_size=batchsize,shuffle=True)
test = ferDataset("./datas/valdata.csv",transform=transform_test)
testloader = DataLoader(dataset=test,batch_size=batchsize)
private_test = ferDataset("./datas/testdata.csv", transform=transform_test)
private_loader = DataLoader(dataset=private_test, batch_size=batchsize)

decay_epoch_start = 60
decay_every = 5
decay_rate = 0.9


losspath = "./losses"
gl = 0
import os
while os.path.exists(os.path.join(losspath,"vgg%d.txt"%gl)):
    gl+=1
lpath = os.path.join(losspath,"vgg%d.txt"%gl)
lf = open(lpath,'w')

def train(epoch):
    model.train()
    import random
    if epoch > decay_epoch_start:
        frac = (epoch-decay_epoch_start) // decay_every
        decay_frac = (decay_rate)**frac
        curlr = lr*decay_frac
        for group in optimizer.param_groups:
            group['lr'] = curlr
    for batchidx, (data, target) in enumerate(dataloader):
        correct = 0
        nl = data.shape[0]
        noise = torch.rand(nl, 3, 44, 44)
        data = torch.add(data, 0.05 * noise)
        if use_cuda:
            data,target = data.cuda(), target.cuda()
        data,target = Variable(data),Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        utils.clip_gradient(optimizer, 0.1)
        optimizer.step()
        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        if batchidx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tCorrec: {: d}'.format(
                epoch, batchidx * len(data), len(dataloader.dataset),
                       100. * batchidx / len(dataloader), loss,correct))
            lf.write('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tCorrec: {: d}\n'.format(
                epoch, batchidx * len(data), len(dataloader.dataset),
                       100. * batchidx / len(dataloader), loss,correct))
def validate(best_acc,save = False):
    model.eval()
    test_loss = 0
    correct = 0
    for bidx,(data,target) in enumerate(testloader):
        bs, ncrops, c, h, w = data.shape
        data = data.view(-1, c, h, w)
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        with torch.no_grad():
            data,target = Variable(data),Variable(target)
            output = model(data)
            output = output.view(bs, ncrops, -1).mean(1)
            test_loss += criterion(output, target)
            pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()
    test_loss /= len(testloader.dataset)
    if(save):
        acc = float(1.0 * correct) / len(testloader.dataset)
        torch.save(model.state_dict(), "./models/add_gvgg2model%.4f.pth"%acc)
        print("Saving in "+"./models/add_gvgg2model%.4f.pth"%acc)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(testloader.dataset),
        100. * correct / len(testloader.dataset)))
    acc = float(1.0 * correct) / len(testloader.dataset)
    lf.write("Acc pub: %.4f\n" % acc)
    if acc>best_acc:
        best_acc = acc
    print("Current best acc: %.4f"%best_acc)
    return best_acc

def private_func(best_acc):
    model.eval()
    test_loss = 0
    correct = 0
    for data,target in private_loader:
        bs, ncrops, c, h, w = data.shape
        data = data.view(-1, c, h, w)
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        with torch.no_grad():
            data,target = Variable(data),Variable(target)
            output = model(data)
            output = output.view(bs, ncrops, -1).mean(1)
            test_loss += criterion(output, target)
            pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()
    test_loss /= len(private_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(private_loader.dataset),
        100. * correct / len(private_loader.dataset)))
    acc = float(1.0 * correct) / len(testloader.dataset)
    lf.write("Acc pri: %.4f\n" % acc)
    if acc>best_acc:
        best_acc = acc
    print("Current best acc: %.4f"%best_acc)
    return best_acc

if __name__ == "__main__":
    best_acc_val = 0
    best_acc_pri = 0
    for i in range(1,epochs+1):
        train(i)
        save = False
        if(i%5==0):
            save=True
        best_acc_val = validate(best_acc_val,save)
        best_acc_pri = private_func(best_acc_pri)
    lf.close()
