import argparse
import os
import random
import yaml
import time
import logging
import pprint

import torch
import scipy.stats as stats
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import numpy as np
from tensorboardX import SummaryWriter
from torch.autograd import grad
from easydict import EasyDict

import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data.sampler import Sampler
from torchvision.transforms import Resize, CenterCrop

from data.train import *
from models.standard import *

print(f'Are you using GPU? --> {torch.cuda.is_available()}')


image_size = 512
batch_size = 4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description='PyTorch Colorization Training')

parser.add_argument('--config', default='experiments/origin/config.yaml')

args = parser.parse_args()
print(args)

with open(args.config) as f:
    config = EasyDict(yaml.load(f))

config.device = device
config.batch_size = 2

print(config.image_size)
CTrans = transforms.Compose([
        Resize(config.image_size, Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.5), (0.5))
    ])

VTrans = transforms.Compose([
    RandomSizedCrop(config.image_size // 4, Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize((0.5), (0.5))
])

def jitter(x):
    ran = random.uniform(0.7, 1)
    return x * ran + 1 - ran

STrans = transforms.Compose([
    Resize(config.image_size, Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Lambda(jitter),
    transforms.Normalize((0.5), (0.5))
])

def mask_gen():
    maskS = config.image_size // 4

    mu, sigma = 1, 0.005
    X = stats.truncnorm((0 - mu) / sigma, (1 - mu) / sigma, loc=mu, scale=sigma)

    mask1 = torch.cat(
        [torch.rand(1, 1, maskS, maskS).ge(X.rvs(1)[0]).float() for _ in range(config.batch_size // 2)], 0)
    mask2 = torch.cat([torch.zeros(1, 1, maskS, maskS).float() for _ in range(config.batch_size // 2)], 0)
    mask = torch.cat([mask1, mask2], 0)

    return mask.to(config.device)

#"F:\3rd-HK2\User-Guild\datasets\illustrations_resized\6.png"
pth_color = "F:\\3rd-HK2\\User-Guild\\datasets\\illustrations_resized\\2.png"
pth_sketch = "F:\\3rd-HK2\\User-Guild\\datasets\\lineart\\2.png"

Cimg = color_loader(pth_color)
Simg = sketch_loader(pth_sketch)
Cimg, Simg = RandomCrop(512)(Cimg, Simg)

Cimg, Vimg, Simg = CTrans(Cimg), VTrans(Cimg), STrans(Simg)

Cimg, Vimg, Simg =  Cimg.to(device), Vimg.to(device), Simg.to(device)

# Cimg target imgs
# Vimg
# Simg sketch imgs

netG = NetG(ngf=config.ngf)

netI = NetI()

mask = mask_gen()
hint = torch.cat((Vimg * mask, mask), 1)

feat_sim = netI(Simg)

fake = netG(Simg, hint, feat_sim)

print(Cimg.shape)

print(hint.shape)


plt.figure(figsize=(10, 10))
#show Cimg
plt.imshow(fake.permute(1, 2, 0).numpy().astype("uint8"))
plt.title("output of generaator")
plt.show()

# #show Vimg
# plt.imshow(Vimg.permute(1, 2, 0).numpy().astype("uint8"))
# plt.title("Vimg")
# plt.show()

# #show Simg
# plt.imshow(Simg.permute(1, 2, 0).numpy().astype("uint8"))
# plt.title("Simg")
# # plt.axis("off")
# plt.show()
# print('done')

# #show hint
# plt.imshow(hint.permute(1, 2, 0).numpy().astype("uint8"))
# plt.title("hint")
# # plt.axis("off")
# plt.show()
# print('done')
