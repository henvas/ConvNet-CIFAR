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





layer1 = torchvision.models.resnet18(pretrained=True).conv1

#print(layer1.weight.data.numpy()[0, :, :, :].shape)

w = layer1.weight.data
print(w.shape)
#img = img.view(7,7,3)
#plt.imshow(img)
#plt.show()

fig = plt.figure(figsize=(10,10))
columns = 8
rows = 8
for i in range(1, columns*rows + 1):
    img = w[i-1].numpy().transpose(1, 2, 0)
    fig.add_subplot(rows, columns, i)
    plt.axis('off')
    plt.subplots_adjust(wspace=0.1, hspace=0.025)
    plt.imshow(img, cmap='gray')
plt.show()

'''
mean = (0.4914, 0.4822, 0.4465)
std = (0.2023, 0.1994, 0.2010)

img = Image.open("boy.jpg")
img = transforms.ToTensor()(img)
img = transforms.Normalize(mean, std)(img)
img = img.view(1, *img.shape)
img = nn.functional.interpolate(img, size=(256, 256))

pred = layer1(img)

bop = pred[0][1].view(128, 128)

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



#plt.imshow(bop.detach().numpy())
#plt.show()

#torchvision.utils.save_image(bop, 'filters.png')
