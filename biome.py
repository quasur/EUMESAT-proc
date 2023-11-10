
#%% =============K MEANS AND SLIC=====================================
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.segmentation import slic
from skimage.color import label2rgb
import argparse


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

def ocean_mask(image):
    
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
    return mask_out#segmentedimage

#%%
#Import mean values for bad pixel interpolation
datapath="C:/Users/adamc/Desktop/bigdata/"
meanimg =np.uint8(np.loadtxt(datapath+"meanimg.np").reshape((3712,3712,3)))
#mask of africa
mask = plt.imread("mask.png")[:,:,0:3]
#array initialisation
biome = np.zeros((3712,3712,23),dtype=np.uint8)
ones = np.ones((3712,3712),dtype=np.uint8)

for i in range(23):
    print(i)
    #read montly img
    img = np.uint8(plt.imread("monthimg/month"+str(i)+".png")*255)
    fiximg = img.copy()
    #set "missing pixels" to the corresponding value in the mean img
    badpx = np.all(fiximg==0,axis=2)
    fiximg[badpx]=meanimg[badpx]

    #use slic to remove the ocean and group biomes
    out1 =slic_image(fiximg,mask)
    groupimg= ocean_mask(out1)

    #Use plane thresholding to classify biomes
    curbiome =np.zeros((3712,3712),dtype=np.uint8)
    desertmask1 = np.any(groupimg[:,:,:]>0,axis=2)
    forestmask =np.logical_and((groupimg[:,:,0]<=groupimg[:,:,1]+15),desertmask1)
    aridmask =  np.logical_and(~forestmask,desertmask1)
    desertmask2 = groupimg[:,:,2]>110
    curbiome[desertmask1]=1
    curbiome[aridmask]=1
    curbiome[forestmask]=2
    curbiome[desertmask2]=1
    biome[:,:,i]=curbiome


#%%
#Create a plot of the relative change in biome over time by lattitude
biomeones = np.ones((3712,3712,23))
biomeones[biome!=0]=0
weights = np.sum(biomeones,axis=1)
weights[weights==0]=1
biomerows = np.sum(biome,axis=1)/weights
biomerowsavg = np.repeat(np.mean(biomerows,axis=1)[:,np.newaxis],23,axis=1)
biomerowsnorm = biomerows-biomerowsavg
plt.figure(dpi=400)
plt.axis("off")
plt.title("Variance of biome through time")
plt.imshow(biomerowsnorm,aspect=23/3712)

#%%
# Create a plot that shows biome variance in space, what regions change the most 
biomechange = np.std(biome,axis=2)
plt.figure(dpi=400)
plt.axis("off")
plt.title("Variance of biome")
plt.imshow(biomechange,aspect=3712/3712)


