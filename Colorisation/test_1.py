#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 16:34:08 2022

@author: oumaimanajib
"""
#%%

import sys
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from skimage import color
import torchvision.models as models

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torchvision.datasets import ImageFolder 

import glob
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    def __init__(self):
        self.imgs_path = "DB_10/"
        file_list = glob.glob(self.imgs_path + "*")
        print(file_list)
        self.data = []
        class_path_1 = file_list[0]
        class_path_2 = file_list[1]
        class_name_1 = class_path_1.split("/")[-1]
        class_name_2 = class_path_2.split("/")[-1]

        for img_path1 in glob.glob(class_path_1+ "/*.png"):
            for img_path2 in glob.glob(class_path_2 + "/*.png"):
                if(img_path1[13:19] == img_path2[15:21]):
                    self.data.append([img_path1, class_name_1, img_path2, class_name_2])
        print(self.data)
        
        self.class_map = {"NIGHT" : 1, "DAY": 0}
        self.img_dim = (416, 416)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_path1, class_name1, img_path2, class_name2 = self.data[idx]
        img1 = cv2.imread(img_path1)
        img2 = cv2.imread(img_path2)

        img1 = cv2.resize(img1, self.img_dim)
        img2 = cv2.resize(img2, self.img_dim)

        class_id1 = self.class_map[class_name1]
        class_id2 = self.class_map[class_name2]


        img_tensor1 = torch.FloatTensor(np.transpose(color.rgb2lab(np.array(img1)), (2, 0, 1)))
        img_tensor2 = torch.FloatTensor(np.transpose(color.rgb2lab(np.array(img2)), (2, 0, 1)))
        img_tensor1.resize(100)
        img_tensor2.resize(100)
        class_id1 = torch.tensor([class_id1])
        class_id2 = torch.tensor([class_id2])

        return img_tensor1, class_id1, img_tensor2, class_id2


num_epochs = 30
batch_size = 5
learning_rate = 1e-3
use_gpu = True

import numpy as np
from skimage import color

import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# converts the PIL image to a pytorch tensor containing an LAB image
def import_image(img):
    return torch.FloatTensor(np.transpose(color.rgb2lab(np.array(img)), (2, 0, 1)))
    
img_transform = transforms.Compose([
    transforms.Lambda(import_image),
    transforms.Resize(100) #resize 1080x1920 ->112x199
])
   # " redefinir la méthode getitem pour avoir deux paires d'images (jour,nuit) " 
train_dataset = CustomDataset() 
#%%
#sortie :    (Tuple[List[str], Dict[str, int]])
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
#%%
test_dataset = ImageFolder('DB_10/NIGHT', transform=img_transform ) 
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
dataiter = iter(train_dataloader)
images, labels = dataiter.next()

class ColorNet(nn.Module):
    def __init__(self, d=128):
        super(ColorNet, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1) # out: 32 x 16 x 16
        self.conv1_bn = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1) # out: 64 x 8 x 8
        self.conv2_bn = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1) # out: 128 x 4 x 4
        self.conv3_bn = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1) # out: 128 x 4 x 4
        self.conv4_bn = nn.BatchNorm2d(128)
        self.conv5 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1) # out: 128 x 4 x 4
        self.conv5_bn = nn.BatchNorm2d(128)
        self.conv6 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1) # out: 128 x 4 x 4
        self.conv6_bn = nn.BatchNorm2d(128)
        self.tconv1 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1) # out: 64 x 8 x 8
        self.tconv1_bn = nn.BatchNorm2d(64)
        self.tconv2 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1) # out: 32 x 16 x 16
        self.tconv2_bn = nn.BatchNorm2d(32)
        self.tconv3 = nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1) # out: 2 x 32 x 32

    def forward(self, input):
        x = F.relu(self.conv1_bn(self.conv1(input)))
        x = F.relu(self.conv2_bn(self.conv2(x)))
        x = F.relu(self.conv3_bn(self.conv3(x)))
        x = F.relu(self.conv4_bn(self.conv4(x)))
        x = F.relu(self.conv5_bn(self.conv5(x)))
        x = F.relu(self.conv6_bn(self.conv6(x)))
        x = F.relu(self.tconv1_bn(self.tconv1(x)))
        x = F.relu(self.tconv2_bn(self.tconv2(x)))
        x = self.tconv3(x)

        return x
    
cnet = ColorNet()
print(cnet)
# %%
device = torch.device("cuda:0" if use_gpu and torch.cuda.is_available() else "cpu")
cnet = cnet.to(device)

num_params = sum(p.numel() for p in cnet.parameters() if p.requires_grad)
print('Number of parameters: %d' % (num_params))
#%%

optimizer = torch.optim.Adam(params=cnet.parameters(), lr=learning_rate)

# set to training mode
cnet.train()

train_loss_avg = []
#%%

print('Training ...')
for epoch in range(num_epochs):
    train_loss_avg.append(0)
    num_batches = 0
    
    for lab_batch, _ in train_dataloader:
        
        lab_batch = lab_batch.to(device)
        
        # apply the color net to the luminance component of the Lab images
        # to get the color (ab) components
        predicted_ab_batch = cnet(lab_batch[:, :, :, :])
        
        # loss is the L2 error to the actual color (ab) components
        loss = F.mse_loss(predicted_ab_batch, lab_batch[:, 1:3, 4:, 1:])
        
        # backpropagation
        optimizer.zero_grad()
        loss.backward()
        
        # one step of the optmizer (using the gradients from backpropagation)
        optimizer.step()
        
        train_loss_avg[-1] += loss.item()
        num_batches += 1
        
    train_loss_avg[-1] /= num_batches
    print('Epoch [%d / %d] average reconstruction error: %f' % (epoch+1, num_epochs, train_loss_avg[-1]))


import matplotlib.pyplot as plt
plt.ion()

fig = plt.figure(figsize=(15, 5))
plt.plot(train_loss_avg)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()

# set to evaluation mode
cnet.eval()
#%%
test_loss_avg, num_batches = 0, 0
for lab_batch, _ in test_dataloader:

    with torch.no_grad():

        lab_batch = lab_batch.to(device)

        # apply the color net to the luminance component of the Lab images
        # to get the color (ab) components
        predicted_ab_batch = cnet(255 - lab_batch[:, 0:1, :, :])
        
        # loss is the L2 error to the actual color (ab) components
        loss = F.mse_loss(predicted_ab_batch, lab_batch[:, 1:3, 4:, 1:])

        test_loss_avg += loss.item()
        num_batches += 1
    
test_loss_avg /= num_batches
print('average loss: %f' % (test_loss_avg))

import numpy as np
from skimage import color, io

import matplotlib.pyplot as plt
plt.ion()
#%%
import torchvision.utils

with torch.no_grad():

    # pick a random subset of images from the test set
    image_inds = np.random.choice(len(test_dataset), 3)
    lab_batch = torch.stack([test_dataset[i][0] for i in image_inds])
    lab_batch = lab_batch.to(device)

    # predict colors (ab channels)
    predicted_ab_batch = cnet(100-lab_batch[:, 0:1, :, :])
    max = lab_batch[:,0:1, 4:, 1:].max()
    predicted_lab_batch = torch.cat([100-lab_batch[:,0:1, 4:, 1:], predicted_ab_batch], dim=1)

    lab_batch = lab_batch.cpu()
    predicted_lab_batch = predicted_lab_batch.cpu()

    # convert to rgb
    rgb_batch = []
    predicted_rgb_batch = []
    for i in range(lab_batch.size(0)):
        rgb_img = color.lab2rgb(np.transpose(lab_batch[i, :, :, :].detach().numpy().astype('float64'), (1, 2, 0)))
        rgb_batch.append(torch.FloatTensor(np.transpose(rgb_img, (2, 0, 1))))
        predicted_rgb_img = color.lab2rgb(np.transpose(predicted_lab_batch[i, :, :, :].detach().numpy().astype('float64'), (1, 2, 0)))
        predicted_rgb_batch.append(torch.FloatTensor(np.transpose(predicted_rgb_img, (2, 0, 1))))

    # plot images
    fig, ax = plt.subplots(figsize=(15, 15), nrows=1, ncols=2)
    ax[0].imshow(np.transpose(torchvision.utils.make_grid(torch.stack(predicted_rgb_batch), nrow=5).numpy(), (1, 2, 0)))
    ax[0].title.set_text('re-colored')
    ax[1].imshow(np.transpose(torchvision.utils.make_grid(torch.stack(rgb_batch), nrow=5).numpy(), (1, 2, 0)))
    ax[1].title.set_text('original')
    plt.show()

