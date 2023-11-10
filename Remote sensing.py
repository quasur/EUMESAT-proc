#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 14:23:00 2023

@author: alex
"""
#%%
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from PIL import Image

mypath = '/Users/alex/Documents/Physics 4th Year/Imaging & date processing/Sat img/'

from os import walk

f = []
for (dirpath, dirnames, filenames) in walk(mypath):
    f.extend(filenames)
    break

size = np.size(filenames)
props = np.zeros((3,size)).astype("object")

for i in range(size):
    if filenames[i][0:4] == "IR16":
        props[0,i]=0
    elif filenames[i][0:4] == "VIS8":
        props[0,i]=1
    elif filenames[i][0:4] == "VIS6":
        props[0,i]=2
    else:
        props[0,i]=3

    props[1,i] = int(filenames[i][29:43])#YYYY mm DD HH MM SS
    props[2,i] = filenames[i]

    
sort = props[:, np.lexsort((props[0,:], props[1, :]))]

#time ordered array of idexing: filter, datetime, filename 

#%%
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.color import label2rgb
from skimage.util import img_as_float
from skimage.color import rgb2gray
from skimage import data 
from skimage.filters import gaussian
from skimage.segmentation import active_contour

import argparse

meanimg = (np.loadtxt('meanimg.np').reshape((3712,3712,3)).copy())

image = meanimg[:, :, 0]

#%%

#Change the image array shape into q,3 rather than x,y,3
pxval=meanimg.reshape((-1))
pxval =np.float32(pxval)

#number of means 
k=10
#Stopping criteria:
#Error threshold
E_Val =0.01
#Max iterations                        
max_iterations = 10000

initial_means = cv2.KMEANS_RANDOM_CENTERS #cv2.KMEANS_PP_CENTERS

#Define the stopping criteria as a single variable to input into the function
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,max_iterations,E_Val)
#calculate the k-means clusters
compactness,labels,(centers)=cv2.kmeans(pxval,k,None,criteria,1,initial_means)

#data formatting required for functions to play nice
centers = np.uint8(centers)
labels=labels.flatten()

#convert pixels to colour of centroid

segmentedimage = centers[labels.flatten()]

#reshape back into original dimension

segmentedimage = segmentedimage.reshape(meanimg.shape)

#%% ==================================KMEANS PLOTTING======================

'''
#plot original image

plt.figure(dpi=400)
plt.imshow(labels.reshape((3712,3712)))
plt.axis("off")
plt.show()
'''
#plot new image

plt.figure(dpi=400)
plt.imshow(segmentedimage)
plt.axis("off")
plt.show()

#%%
import glob 
import numpy.ma as ma
from PIL import Image, ImageFile 

red_indices = np.argwhere(sort[0, :]==0)
green_indices = np.argwhere(sort[0, :]==1)
blue_indices = np.argwhere(sort[0, :]==2)

def pixelwise(sort, mypath, indices):
    
    pixval_array = np.zeros((3712, 3712))
    no_it = 0
    
    for i in indices:
        

        fil_nm = sort[2, i]
            
        file_read = os.path.join(mypath, fil_nm[0])
            
        img = plt.imread(file_read)
        
        pixval_array += img
        
        no_it+=1
        
    print('Channel complete')    
    return pixval_array, no_it
    

def multi_img(ind):

    pixval_red, no_it = pixelwise(sort, mypath, red_indices[ind:ind+73])
    pixval_green, no_it = pixelwise(sort, mypath, green_indices[ind:ind+73])
    pixval_blue, no_it = pixelwise(sort, mypath, blue_indices[ind:ind+73])
    
    red_img = pixval_red/no_it
    green_img = pixval_green/no_it
    blue_img = pixval_blue/no_it
    
    comb_image_1 = np.dstack((red_img, green_img)) 
    comb_image_2 = np.dstack((comb_image_1, blue_img))
    
    print('image_complete')
    return comb_image_2
 
Mean_1 = multi_img(0)
Mean_2 = multi_img(73)
Mean_3 = multi_img(146)
Mean_4 = multi_img(219)
Mean_5 = multi_img(292)
Mean_6 = multi_img(365)
Mean_7 = multi_img(438)
Mean_8 = multi_img(511)
Mean_9 = multi_img(584)
Mean_10 = multi_img(657)

#%%

Combined_array = np.stack((Mean_1, Mean_2, Mean_3, Mean_4, Mean_5, Mean_6, Mean_7, Mean_8, Mean_9, Mean_10), axis = -1)
    
#%%=============================SLIC=================================

mask = plt.imread('mask.png')[:, :, 0:3]
plt.imshow(np.uint8(mask*meanimg))

image = mask*meanimg
#define constants
numSp = 35 #number of superpixels
c= 1  #colour weight
sigma = .5 #factor for pre-processing gaussian blur

#calculate superpixel labels
segments =slic(image, n_segments=numSp,compactness=c,sigma=sigma,enforce_connectivity=False,max_size_factor=3,channel_axis=-1)
#post processing step changes all values in a superpixel to mean value
out1 = label2rgb(segments, image, kind="avg",bg_label=0)

#%%=========================SLIC PLOTTING===========================

#plot the images
plt.figure(dpi=400)
plt.imshow(np.uint8(out1))
plt.axis("off")
plt.title('Output from SLIC')
plt.show()

ret, threshold = cv2.threshold(out1, 90, 255, cv2.THRESH_BINARY)

plt.figure(dpi=400)
plt.imshow(np.uint8(out1), cmap='Greys_r')
plt.axis("off")
plt.title('Thresholded slic image')
plt.show()


plt.figure(dpi=400)
plt.imshow(np.uint8(mask*meanimg), cmap='Greys')
plt.title('Target image')
plt.axis("off")
plt.show()

#%%

slic_mask = cv2.inRange(out1[:, :, 0],0, 50)
mask_ind = np.argwhere(slic_mask==255)
mask_out = np.copy(out1)

for i in range(0, len(mask_ind)):
    
    ind_x = mask_ind[i,0]
    ind_y = mask_ind[i,1]
    
    mask_out[ind_x, ind_y] = 0
    
plt.imshow(np.uint8(mask_out), cmap='binary')

#%%
'''

Kmeans on SLIC

'''

test = np.uint8(mask_out)
#Change the image array shape into q,3 rather than x,y,3
pxval=test.reshape((-1))
pxval =np.float32(pxval)

#number of means 
k= 8
#Stopping criteria:
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

plt.imshow(test)
plt.figure(dpi=400)
plt.imshow(segmentedimage)
plt.axis("off")
plt.show()

#%%

'''

Colourspace plotting

'''

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import colors

#removing duplicates

r_flat = np.ndarray.flatten((out1[:, :, 0]))
vis8_flat = np.ndarray.flatten((out1[:, :, 1]))
vis6_flat = np.ndarray.flatten((out1[:, :, 2]))

r_rduced = np.unique(np.uint8(r_flat))
vis8_rduced = np.unique(np.uint8(vis8_flat))
vis6_rduced = np.unique(np.uint8(vis6_flat))

spix_val = np.stack((r_rduced, vis8_rduced, vis6_rduced))

# Extract RGB components
r = spix_val[0, :]
g = spix_val[1, :]
b = spix_val[2, :]

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, projection="3d")

col_val = []

for i in range(np.size(spix_val[1])):
    
    pixel_colors = spix_val[:,i]
    pixel_colors = pixel_colors/255
    
    print(pixel_colors)

    pixel_colors = pixel_colors.tolist()
    
    col_val.append(pixel_colors)

ax.scatter(r, g, b, '' ,facecolors=col_val, s=50)
    
# Set axis labels
ax.set_xlabel('Red')
ax.set_ylabel('Green')
ax.set_zlabel('Blue')

# Show the plot
plt.show()

plt.figure(dpi=400)
plt.imshow(np.uint8(out1), cmap='Greys_r')
plt.axis("off")
plt.title('Thresholded slic image')
plt.show()

#%%

'''
NDVI sensing
'''

meanimg = (np.loadtxt('meanimg.np').reshape((3712,3712,3)).copy())
#%%
from numpy import *

NDVI_image = (meanimg[:, :, 1]-meanimg[:, :, 2])/(meanimg[:, :, 1]+meanimg[:, :, 2])
where_are_NaNs = isnan(NDVI_image)
NDVI_image[where_are_NaNs] = 0
#%%
plt.imshow(NDVI_image)


#%%