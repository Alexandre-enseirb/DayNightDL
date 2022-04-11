#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 11 10:40:24 2022

@author: oumaimanajib
"""

import sys
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from skimage import color
import torchvision.models as models
from scipy.linalg import toeplitz
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torchvision.datasets import ImageFolder 
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from matplotlib import cm

#%%

def import_image(img):
    return np.transpose(color.rgb2lab(np.array(img)), (0, 1, 2))



img_transform = transforms.Compose([
    transforms.Resize(112),
    transforms.Lambda(import_image)
])
    
#%% night
    
dataset_n = ImageFolder('DB_10/NIGHT', transform=img_transform ) 
img1 = dataset_n[0][0]
print(img1.shape)


#%%day 
dataset_d = ImageFolder('DB_10/DAY', transform=img_transform ) 
img2 = dataset_d[0][0]
print(img2.shape)
#%% transform


fig, axs = plt.subplots(2)
fig.suptitle('Vertically stacked subplots')
axs[0].imshow(img1[:,:,1]-50,cmap="Greys")
axs[0].set_title("image de nuit")
axs[1].imshow(img2[:,:,1], cmap='Greys')
axs[1].set_title("image de jour")

#%% traitement 
#dans cette partie, on considère l'image comme un processus aléatoire discret 

#matrice d'autocorrélation sur une partie de l'image
vecteur1 = np.ravel(img1[50:100,50:100,1])
acf = np.convolve(vecteur1,np.conj(vecteur1)[::-1]) # using Method 2 to compute Auto-correlation sequence
Rxx=acf[2:]; # R_xx(0) is the center element
R_d1 = toeplitz(Rxx,np.hstack((Rxx[0], np.conj(Rxx[1:]))))


#%%
vecteur2 = np.ravel(img2[50:100,50:100,1])
acf = np.convolve(vecteur2,np.conj(vecteur2)[::-1]) # using Method 2 to compute Auto-correlation sequence
Rxx=acf[2:]; # R_xx(0) is the center element
R_d2 = toeplitz(Rxx,np.hstack((Rxx[0], np.conj(Rxx[1:]))))


#%% affichage de la matrice d'autocorrelation 

fig, ax = plt.subplots(2)
max_val, min_val = int(R_d1.max()), int(R_d1.min())
ax[0].matshow(R_d1.astype(int), cmap=plt.cm.Blues)
ax[1].matshow(R_d2.astype(int), cmap=plt.cm.Blues)

#les images de nuit ne sont pas très correlés 
#%%
fig , ax = plt.subplots(2)
ax[0].hist(vecteur1,  bins = 10, color = 'yellow',
            edgecolor = 'red')
plt.xlabel('valeurs')
plt.ylabel('nombres')
plt.title('histogramme nuit')
ax[1].hist(vecteur2,  bins = 10, color = 'yellow',
            edgecolor = 'red')
plt.xlabel('valeurs')
plt.ylabel('nombres')
plt.title('histogramme jour')

