#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 14:23:00 2023

@author: alex
"""
#%%
import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.segmentation import slic
from skimage.color import label2rgb



#loading of images 

meanimg = (np.loadtxt('meanimg.np').reshape((3712,3712,3)).copy())
mask = plt.imread('mask.png')[:, :, 0:3]

#%%============================DEFINING FUNCTIONS====================

def slic_image(image, mask):
    
    #mask the image
    test = mask*image
    
    #define constants
    numSp = 35 #number of superpixels
    c= 1  #colour weight
    sigma = .5 #factor for pre-processing gaussian blur

    #calculate superpixel labels
    segments =slic(test, n_segments=numSp,compactness=c,sigma=sigma,enforce_connectivity=False,max_size_factor=3,channel_axis=-1)
    
    #post processing step changes all values in a superpixel to mean value
    out1 = np.uint8(label2rgb(segments, test, kind="avg",bg_label=0))
    
    return out1

def kmeans_image(image):
    
    #thresholding based on red channel value, might have to tweak
    
    slic_mask = cv2.inRange(image[:, :, 0],0, 50)
    mask_ind = np.argwhere(slic_mask==255)
    mask_out = np.copy(image)
    
    #sets water indices to zero
    for i in range(0, len(mask_ind)):
        
        ind_x = mask_ind[i,0]
        ind_y = mask_ind[i,1]
        
        mask_out[ind_x, ind_y] = 0
        
    #defines target image    
    test = np.uint8(mask_out)
    
    #Change the image array shape into q,3 rather than x,y,3
    pxval=test.reshape((-1))
    pxval =np.float32(pxval)

    #number of means 
    k= 8

    #Error threshold
    E_Val = 0
    #Max iterations                        
    max_iterations = 10000


    initial_means = cv2.KMEANS_RANDOM_CENTERS #cv2.KMEANS_PP_CENTERS

    #Define the stopping criteria as a single variable to input into the function
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,max_iterations,E_Val)
    #calculate the k-means clusters
    compactness,labels,(centers)=cv2.kmeans(pxval, k,None,criteria,5,initial_means)

    #data formatting required for functions to play nice
    centers = np.uint8(centers)
    labels=labels.flatten()

    #convert pixels to colour of centroid

    segmentedimage = centers[labels.flatten()]

    #reshape back into original dimension

    segmentedimage = segmentedimage.reshape(meanimg.shape)
    
    return segmentedimage


#%%

#returns desired images
output_image = slic_image(meanimg, mask)
#%%

fig = plt.figure(dpi=300)

ax1 = fig.add_subplot(1, 2, 1)
ax2 = fig.add_subplot(1, 2, 2)

ax1.imshow(np.uint8(meanimg))
ax2.imshow(output_image)

ax1.set_title('Input test image')
ax1.set_xticks([])
ax1.set_yticks([])

ax2.set_title('Output from SLIC')
ax2.set_xticks([])
ax2.set_yticks([])

#%%

k_means_image = kmeans_image(output_image)
plt.imshow(k_means_image)

