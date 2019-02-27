import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from dataloaders import load_cifar10
import torchvision
import glob
from PIL import Image
import PIL
from torchvision import transforms

mean = (0.4914, 0.4822, 0.4465)
std = (0.2023, 0.1994, 0.2010)

model = torchvision.models.resnet18(pretrained=True)

img = Image.open("boy.jpg")
img = transforms.ToTensor()(img)
img = transforms.Normalize(mean, std)(img)
img = img.view(1, *img.shape)
img = nn.functional.interpolate(img, size=(256, 256))
#print(img.shape)



modules = list(model.children())[:-2]

first = True
for i in modules:
    if first:
        pred = i(img)
        first = False
    else:
        pred = i(pred)


fig = plt.figure(figsize=(8,8))
columns = 5
rows = 2
for i in range(1, columns*rows + 1):
    img = pred[0][i].view(8,8)
    fig.add_subplot(rows, columns, i)
    plt.axis('off')
    plt.subplots_adjust(wspace=0.1, hspace=0.025, bottom=0.43, left=0.13, top=0.88)
    plt.imshow(img.detach().numpy(), cmap='gray')
plt.show()


#print(modules)



'''
fig = plt.figure(figsize=(8,8))
columns = 5
rows = 2
for i in range(1, columns*rows + 1):
    img = pred[0][i].view(128,128)
    fig.add_subplot(rows, columns, i)
    plt.axis('off')
    plt.subplots_adjust(wspace=0.1, hspace=0.025, bottom=0.43, left=0.13, top=0.88)
    plt.imshow(img.detach().numpy(), cmap='gray')
plt.show()'''
