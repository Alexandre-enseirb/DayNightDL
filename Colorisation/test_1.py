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
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from torchvision.datasets import ImageFolder

import glob
import cv2
import numpy as np


num_epochs = 40
batch_size = 2
learning_rate = 1e-2
use_gpu = True

# set the following var to True to run faster (/!\ BREAKS IF NO WEIGHTS STORED)
use_saved = False

def imshow(tensor, title=None, transpose=True):
    """
    converts a tensor into a PIL image and displays it
    Tensor has to be shaped [1,c,h,w] or [c,h,w]
    """

    # reducing dimension
    if len(tensor.shape)==4:
        tensor = tensor[0,:,:,:]
    if tensor.requires_grad:
        tensor = tensor.detach()
    img = tensor.numpy()
    print("Shape: {}".format(img.shape))
    if transpose:
        img = np.transpose(img,(1,2,0))

    plt.figure()
    plt.imshow(img)
    if not title is None:
        plt.title(title)



def import_image(img):
    return torch.FloatTensor(np.transpose(np.array(img)/255, (2, 0, 1)))

img_transform = transforms.Compose([
    transforms.Lambda(import_image),
    transforms.Resize(100) #resize 1080x1920 ->112x199
])

class CustomDataset(Dataset):
    def __init__(self):
        self.imgs_path = "../DB_10/"
        file_list = glob.glob(self.imgs_path + "*")
        print(file_list)
        self.data = []
        class_path_1 = file_list[0]
        class_path_2 = file_list[1]
        class_name_1 = class_path_1.split("/")[-1]
        class_name_2 = class_path_2.split("/")[-1]
        L = []
        for img_path1 in glob.glob(class_path_1+ "/*.png"):
            for img_path2 in glob.glob(class_path_2 + "/*.png"):
                start1 = img_path1.find('_d') + 3
                end1 = img_path1.find('.png', start1)
                start2 = img_path2.find('_d') + 3
                end2 = img_path2.find('.png', start2)
                if(img_path1[start1:end1] == img_path2[start2:end2] and (img_path1[start1:end1] not in L)):
                    L.append(img_path1[start1:end1])
                    self.data.append([img_path1, class_name_1, img_path2, class_name_2])
        self.data = tuple(self.data)
        print(self.data)

        self.class_map = {"NIGHT" : 1, "DAY": 0}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path1, class_name1, img_path2, class_name2 = self.data[idx]
        img1 = cv2.imread(img_path1)
        img2 = cv2.imread(img_path2)
        class_id1 = self.class_map[class_name1]
        class_id2 = self.class_map[class_name2]
        img_tensor1 = torch.tensor(img1)
        img_tensor2 = torch.tensor(img2)
        img_tensor1 = img_transform(img_tensor1)
        img_tensor2 = img_transform(img_tensor2)
        class_id1 = torch.tensor([class_id1])
        class_id2 = torch.tensor([class_id2])

        return img_tensor1, class_id1, img_tensor2, class_id2


class CustomDataset1(Dataset):
    def __init__(self):

        self.imgs_path = "../DB_10/"

        file_list = glob.glob(self.imgs_path + "*")
        self.data = []
        class_path_1 = file_list[0]
        class_path_2 = file_list[1]
        class_name_1 = class_path_1.split("/")[-1]
        class_name_2 = class_path_2.split("/")[-1]
        L = []
        for img_path1 in glob.glob(class_path_1+ "/*.png"):
            for img_path2 in glob.glob(class_path_2 + "/*.png"):
                start1 = img_path1.find('town') + 3
                end1 = img_path1.find('.png', start1)
                start2 = img_path2.find('town') + 3
                end2 = img_path2.find('.png', start2)
                print(img_path2[start2:end2], img_path1[start1:end1])

                if(img_path1[start1:end1] == img_path2[start2:end2] and (img_path1[start1:end1] not in L)):
                    L.append(img_path1[start1:end1])
                    self.data.append([img_path1, class_name_1, img_path2, class_name_2])
        self.data = tuple(self.data)
        print(self.data)

        self.class_map = {"NIGHT" : 1, "DAY": 0}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path1, class_name1, img_path2, class_name2 = self.data[idx]

        # loads images as (B,G,R)
        img1 = cv2.imread(img_path1)
        img2 = cv2.imread(img_path2)

        # swap as (R,G,B)
        img1[:,:,[0,2]] = img1[:,:,[2,0]]
        img2[:,:,[0,2]] = img2[:,:,[2,0]]


        class_id1 = self.class_map[class_name1]
        class_id2 = self.class_map[class_name2]
        img_tensor1 = torch.tensor(img1)
        img_tensor2 = torch.tensor(img2)
        img_tensor1 = img_transform(img_tensor1)
        img_tensor2 = img_transform(img_tensor2)
        class_id1 = torch.tensor([class_id1])
        class_id2 = torch.tensor([class_id2])

        return img_tensor1, class_id1, img_tensor2, class_id2

learning_rate = 1e-3
use_gpu = True

import numpy as np
from skimage import color

import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# converts the PIL image to a pytorch tensor containing an LAB image

   # " redefinir la méthode getitem pour avoir deux paires d'images (jour,nuit) "
#train_dataset = CustomDataset()
#%%
#sortie :    (Tuple[List[str], Dict[str, int]])
#train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

#%%
#dataiter = iter(train_dataloader)
#image1, label1, image2, label2 = dataiter.next()

#%%
test_dataset = CustomDataset1()
#test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

#%%

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

if not use_saved:
    print('Training ...')
    for epoch in range(num_epochs):
        train_loss_avg.append(0)
        num_batches = 0

        for lab_batch_day, label1 , lab_batch_night, label2 in train_dataloader:

            lab_batch_night = lab_batch_night.to(device)
            lab_batch_day = lab_batch_day.to(device)
            # apply the color net to the luminance component of the Lab images
            # to get the color (ab) components
            predicted_batch = cnet(lab_batch_night[:, :, :, :])

            # loss is the L2 error to the actual color (ab) components
            loss = F.mse_loss(predicted_batch, lab_batch_day[:, :, 4:, 1:])

            # backpropagation
            optimizer.zero_grad()
            loss.backward()

            # one step of the optmizer (using the gradients from backpropagation)
            optimizer.step()

            train_loss_avg[-1] += loss.item()
            num_batches += 1

        train_loss_avg[-1] /= num_batches
        print('Epoch [%d / %d] average reconstruction error: %f' % (epoch+1, num_epochs, train_loss_avg[-1]))
else:
    print("Loading...")
    cnet.load_state_dict(torch.load("cnet.pt"))

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
for lab_batch1, day, lab_batch2, night in test_dataloader:

    with torch.no_grad():

        lab_batch2 = lab_batch2.to(device)
        lab_batch1 = lab_batch1.to(device)
        # apply the color net to the luminance component of the Lab images
        # to get the color (ab) components
        predicted_ab_batch = cnet(lab_batch2[:, :, :, :])

        # loss is the L2 error to the actual color (ab) components
        loss = F.mse_loss(predicted_ab_batch, lab_batch1[:, :, 4:, 1:])

        test_loss_avg += loss.item()
        num_batches += 1

test_loss_avg /= num_batches
print('average loss: %f' % (test_loss_avg))

#%%
import numpy as np
from skimage import color, io

import matplotlib.pyplot as plt
#plt.ion()
#%%
import torchvision.utils

print("Saving weights")
torch.save(cnet.state_dict(), "cnet.pt")



with torch.no_grad():
    # quick test
    im = test_dataset[0][0]

    imshow(im, title="Before transform")

    im2 = cnet(im.unsqueeze(0))

    imshow(im2, title="After transform")

    plt.show()
    # pick a random subset of images from the test set
    """ PREVIOUS VERSION. UNCOMMENT TO TEST. MIGHT BREAK.
    image_inds = np.random.choice(len(test_dataset), 3)
    print(image_inds)

    lab_batch = torch.stack([test_dataset[i][2] for i in image_inds])
    lab_batch = lab_batch.to(device)

    # predict colors (ab channels)
    predicted_batch = cnet(lab_batch[:, :, :, :])

    lab_batch = lab_batch.cpu()
    predicted_batch = predicted_batch.cpu()

    # convert to rgb
    rgb_batch = []
    predicted_rgb_batch = []
    for i in range(lab_batch.size(0)):
        rgb_img = color.lab2rgb(np.transpose(lab_batch[i, :, :, :].detach().numpy().astype('float64'), (1, 2, 0)))
        rgb_batch.append(torch.FloatTensor(np.transpose(rgb_img, (2, 0, 1))))
        predicted_rgb_img = color.lab2rgb(np.transpose(predicted_batch[i, :, :, :].detach().numpy().astype('float64'), (1, 2, 0)))
        predicted_rgb_batch.append(torch.FloatTensor(np.transpose(predicted_rgb_img, (2, 0, 1))))

    # plot images
    fig, ax = plt.subplots(figsize=(15, 15), nrows=1, ncols=2)
    ax[0].imshow(np.transpose(torchvision.utils.make_grid(torch.stack(predicted_rgb_batch), nrow=5).numpy(), (1, 2, 0)))
    ax[0].title.set_text('re-colored')
    ax[1].imshow(np.transpose(torchvision.utils.make_grid(torch.stack(rgb_batch), nrow=5).numpy(), (1, 2, 0)))
    ax[1].title.set_text('original')
    plt.savefig("fig.png")
    """
