#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Code pour observer l'extraction de features par la methode des forward_hooks.
Ce programme utilise un réseau resnet18 plutôt qu'un VGG car il est plus simple d'ajouter un hook pour chaque couche convolutive de resnet18 (elles sont mieux séparées).

PENSEZ A MODIFIER LES VARIABLES img_dir ET img_file.

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# directory and filename
img_dir = None
img_filename = None

# set to False for faster init, set to True for pretrained weights
pretrained=False

features_blobs=[]

def imshow(img):
    plt.figure()
    plt.imshow(img)
    plt.show()
    
def hook_feature(module, inpt, outpt):
    features_blobs.append(outpt.data.numpy())

def mosaique(array, width=512):
    """
    transforme un m*n*p-array en 3*h*w image
    Forme des lignes de 10 images
    """
    [m,n,p] = array.shape
    imgs = []
    i=0
    while i+3 < m:
        imgs.append(array[i:i+3,:,:])
        i+=1
    
    Nimgs = len(imgs)
    print("Extracted {} images".format(len(imgs)))
    Ncols    = 10
    Nlines   = (Nimgs//Ncols)+1
    lenCols  = Ncols * n
    lenLines = Nlines * p
    
    Iout     = np.zeros((3, lenCols,lenLines))
    print("Created ({}x{}) grid. (total: {} x {} px)"
          .format(Ncols,Nlines, lenCols, lenLines))
    
    ptrLines = 0
    ptrCols  = 0
    for img in imgs:
        Iout[:, 
             ptrCols*n:(ptrCols+1)*n, 
             ptrLines*p:(ptrLines+1)*p] = img[:,:,:]
        ptrCols+=1
        if ptrCols==10:
            ptrCols=0
            ptrLines+=1
    
    return np.transpose(Iout, (2,1,0))

def main(imgfile, pretrained=False):
    """
    
    Args:
        - imgfile: l'image qui va servir à l'entrainement
    
    Return:
        None
    """
    
    #%% Création des modèles
    resnet = torchvision.models.resnet18(pretrained=True)
    
    
    # Chargement de l'image
    img = Image.open(imgfile)
    #imshow(img)
    img_np = np.array(img)                          # conversion en np.array
    img_torch = torch.tensor(img_np)                # conversion en torch.Tensor
    img_torch = torch.transpose(img_torch, 0,2)     # dim: 4*1920*1080
    img_torch = img_torch[0:3,:,:]                  # dim: 3*1920*1080
    img_torch = torch.unsqueeze(img_torch, dim=0)   # dim: 1*3*1920*1080
    img_torch_f = img_torch/255                     # converts to float
    print(img_torch.shape)
    
    
    #%% Création des "hooks"
    
    # récupération des couches convolutives
    resnet_layers = ['layer1','layer2','layer3','layer4']
    
    
 
        
    # déclaration des hooks
    
    
    for layer in resnet_layers:
        resnet._modules.get(layer).register_forward_hook(hook_feature)
    
    #%% Extraction des features
    
    resnet(img_torch_f)
    
    # dimensions pour redimensionner
    heights = [0, 2048, 1024, 1024, 512]
    widths  = [0, 512, 256, 128 , 64]
    
    
    i=0
    for feature in features_blobs:
        feature = feature[0,:,:,:]
        feature = mosaique(feature)   
        imshow(feature)
            #imshow(np.transpose(feature[0:3,:,:], (2,1,0)))
        i+=1
        
    return features_blobs
    
if __name__ == '__main__':
    if img_dir is None or img_filename is None:
        raise Exception("Veuillez spécifier un dossier et une image")
    tmp = main(img_dir+img_filename, pretrained)

