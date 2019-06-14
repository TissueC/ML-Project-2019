from __future__ import print_function
from advertorch.context import ctx_noparamgrad_and_eval
from vgg_19_net import *
from dataset import *
import torchvision.transforms as transforms
from advertorch_examples.utils import _imshow
import numpy as np

def get_cls(pred):
    return pred.data.max(1, keepdim=True)[1].cpu().numpy()

def showfig():
    pass

if __name__ == '__main__':
    modelroot = "vgg7125"

    #load model
    torch.manual_seed(0)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    model = VGG("VGG19")
    model.load_state_dict(torch.load("./models/%s.pth"%modelroot))

    model.to(device)
    model.eval()

    cut_size = 44
    batch_size = 5
    transform_train = transforms.Compose([
        transforms.RandomCrop(cut_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # normTransform
    ])

    transform_test = transforms.Compose([
        transforms.TenCrop(cut_size),
        transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
        # normTransform
    ])

    #load data
    datas = ferDataset("./datas/training.csv", transform=transform_train)
    dataloader = DataLoader(dataset=datas, batch_size=batch_size, shuffle=True)

    for cln_data, true_label in dataloader:
        break
    cln_data, true_label = cln_data.to(device), true_label.to(device)

    #attack
    """
    from advertorch.attacks import LinfPGDAttack

    adversary = LinfPGDAttack(model, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=0.15,
    nb_iter=40, eps_iter=0.01, rand_init=True, clip_min=0.0, clip_max=1.0,
    targeted=False)
    """

    from advertorch.attacks import GradientSignAttack
    import time
    t = time.time()
    adversary = GradientSignAttack(model,loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=0.02,
    clip_min=0.0, clip_max=1.0,targeted=False)

    adv_untarget = adversary.perturb(cln_data, true_label)
    target = torch.ones_like(true_label) * 3
    adversary.targeted = True
    adv_targeted = adversary.perturb(cln_data, target)

    pred_cln = get_cls(model(cln_data))
    pred_untarget = get_cls(model(adv_untarget))
    pred_target = get_cls(model(adv_targeted))

    import matplotlib.pyplot as plt
    plt.figure(figsize=(10,8))
    for ii in range(batch_size):
        plt.subplot(3, batch_size, ii+1)
        _imshow(cln_data[ii])
        plt.title("clean \n pred: {}".format(pred_cln[ii]))
        plt.subplot(3, batch_size, ii + 1 + batch_size)
        _imshow(adv_untarget[ii])
        plt.title("untargeted \n adv \n pred: {}".format(
            pred_untarget[ii]))
        plt.subplot(3, batch_size, ii + 1 + batch_size * 2)
        _imshow(adv_targeted[ii])
        plt.title("targeted to 3 \n adv \n pred: {}".format(
            pred_target[ii]))
    plt.tight_layout()
    plt.show()