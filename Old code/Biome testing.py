#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 13:28:39 2023

@author: alex
"""


#%% =============K MEANS AND SLIC=====================================
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.color import label2rgb
from skimage.util import img_as_float
from skimage.color import rgb2gray
from skimage import data 
from skimage.filters import gaussian
from skimage.segmentation import active_contour
import argparse

#%%

def slic_image(image, mask):
    
    #mask the image
    test = mask*image
    
    #define constants
    numSp = 10 #number of superpixels
    c= 1  #colour weight
    sigma = .5 #factor for pre-processing gaussian blur

    #calculate superpixel labels
    segments =slic(test, n_segments=numSp,compactness=c,sigma=sigma,enforce_connectivity=False,max_size_factor=3,channel_axis=-1)
    #post processing step changes all values in a superpixel to mean value
    out1 = np.uint8(label2rgb(segments, test, kind="avg",bg_label=0))
    
    return out1

def kmeans_image(image):
    
    #thresholding based on red channel value, might have to tweak
    slic_mask = cv2.inRange(image[:, :, 0],0, 85)
    mask_ind = np.argwhere(slic_mask==255)
    mask_out = np.copy(image)
    
    #sets water indices to zero
    for i in range(0, len(mask_ind)):
        
        ind_x = mask_ind[i,0]
        ind_y = mask_ind[i,1]
        
        mask_out[ind_x, ind_y] = 0
        
    #defines target image    
    test = np.uint8(mask_out)
    return mask_out#segmentedimage

#%%===============BADPX INTERPOLATION==============================
datapath="C:/Users/adamc/Desktop/bigdata/"
meanimg =np.uint8(np.loadtxt("meanimg.np").reshape((3712,3712,3)))
#%%
mask = plt.imread("mask.png")[:,:,0:3]

biome = np.zeros((3712,3712,23),dtype=np.uint8)
ones = np.ones((3712,3712),dtype=np.uint8)

for i in range(23):
    print(i)
    img = np.uint8(plt.imread("month"+str(i)+".png")*255)
    fiximg = img.copy()
    #set "missing pixels" to the corresponding value in the mean img
    badpx = np.all(fiximg==0,axis=2)
    fiximg[badpx]=meanimg[badpx]


    out1 =slic_image(fiximg,mask)
    groupimg= kmeans_image(out1)

    curbiome =np.zeros((3712,3712),dtype=np.uint8)
    desertmask1 = np.any(groupimg[:,:,:]>=1,axis=2)
    forestmask =np.logical_and((groupimg[:,:,1]+15 )>(groupimg[:,:,0]),desertmask1)
    aridmask =  np.logical_and((groupimg[:,:,1]+15)<(groupimg[:,:,0]),desertmask1)
    desertmask2 = groupimg[:,:,2]>100
    curbiome[desertmask1]=1
    curbiome[aridmask]=1
    curbiome[forestmask]=2
    curbiome[desertmask2]=1
    biome[:,:,i]=curbiome

#%%
biomeones = np.ones((3712,3712,23))
biomeones[biome!=0]=0
weights = np.sum(biomeones,axis=1)
biomerows = np.sum(biome,axis=1)/weights
biomerowsavg = np.repeat(np.mean(biomerows,axis=1)[:,np.newaxis],23,axis=1)

newbiomerows1 = biomerows[:, 0:7]
newbiomerows2 = biomerows[:, 9:23]

#%%
newbiomesrows = np.append(newbiomerows1, newbiomerows2, axis=1)
#%%
biomerowsstd = np.repeat(np.std(newbiomesrows,axis=1)[:,np.newaxis],21,axis=1)
biomerowsnorm = biomerows - biomerowsavg
#plt.imshow(biomerowsnorm,aspect=23/3712)
#plt.imshow(biomerowsavg, aspect=23/3712)
#%%

plt.figure()
std_values = (biomerowsstd[:, 0]).T
x = np.linspace(0, 3712, 3712)

plt.plot(x, std_values, linewidth=0.8)
plt.xlabel('Latitude')
plt.ylabel('Standard deviation')
plt.title('Detection of region changes')

#%%
