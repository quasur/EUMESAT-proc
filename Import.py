#%%=====================CREATE STORTED FILES=============================
import numpy as np
import os

mypath = "C:/Users/adamc/Desktop/images/"
datapath="C:/Users/adamc/Desktop/bigdata/"

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

sort =props[:,np.lexsort((props[0,:],props[1,:]))]

#time ordered array of idexing: filter, datetime, filename 
numimg = int(size/3)
#%%==================PRINT COMPOUND IMAGE====================================
import matplotlib.pyplot as plt
%matplotlib inline

indx = 0
green = plt.imread(mypath+sort[2,indx+2])
red = plt.imread(mypath+sort[2,indx+1])
blue = plt.imread(mypath+sort[2,indx])


compound = np.array([red,green,blue]).transpose((1,2,0))
"""
plt.figure(dpi=100)
plt.imshow(compound)
plt.axis("off")
plt.show()
print(type(compound[1500,1500,1]))
"""
#%%===========================COLOURSPACE PLOTS====================
plt.figure(dpi=400)
plt.scatter(red[1500:2000,:].ravel(),green[1500:2000,:].ravel(),marker=',',s=1,alpha=0.01)
plt.xlabel("red")
plt.ylabel("green")
plt.show()
plt.figure(dpi=400)
plt.scatter(red[1500:2000,:].ravel(),blue[1500:2000,:].ravel(),marker=',',s=1,alpha=0.01)
plt.xlabel("red")
plt.ylabel("blue")
plt.show()
plt.figure(dpi=400)
plt.scatter(green[:,:].ravel(),blue[:,:].ravel(),marker=',',s=1,alpha=0.01)
plt.xlabel("green")
plt.ylabel("blue")
plt.show()

#%% =============K MEANS AND SLIC=====================================
import cv2
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.color import label2rgb
from skimage.util import img_as_float
from skimage.color import rgb2gray
from skimage import data 
from skimage.filters import gaussian
from skimage.segmentation import active_contour

import argparse

image = #IMAGE GOES HERE

#Change the image array shape into q,3 rather than x,y,3
pxval=image.reshape((-1,3))
pxval =np.float32(pxval)

#number of means 
k=20
#Stopping criteria:
#Error threshold
E_Val =0.01
#Max iterations                        
max_iterations = 10000

initial_means = cv2.KMEANS_RANDOM_CENTERS#cv2.KMEANS_PP_CENTERS

#Define the stopping criteria as a single variable to input into the function
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,max_iterations,E_Val)
#calculate the k-means clusters
compactness,labels,(centers)=cv2.kmeans(pxval,k,None,criteria,1,initial_means)

#data formatting required for functions to play nice
centers = np.uint8(centers)
labels=labels.flatten()

#convert pixels to colour of centroid
segmentedimage = centers[labels.flatten()]
#reshape back into original dimention
segmentedimage = segmentedimage.reshape(image.shape)
#%% ==================================KMEANS PLOTTING======================
#plot original image
plt.figure(dpi=400)
plt.imshow(labels.reshape((3712,3712)))
plt.axis("off")
plt.show()
#plot new image
plt.figure(dpi=400)
plt.imshow(segmentedimage)
plt.axis("off")
plt.show()
#%%=============================SLIC=================================
#define constants
numSp =20 #number of superpixels
c=0.1      #colour weight
sigma =0 #factor for pre-processing gaussian blur
image = compound1
#calculate superpixel labels
segments =slic(image,n_segments=numSp,compactness=c,sigma=sigma,enforce_connectivity=False,max_size_factor=3)
#post processing step changes all values in a superpixel to mean value
out1 = label2rgb(segments,image,kind="avg",bg_label=0)
#%%=========================SLIC PLOTTING===========================
#plot the images
plt.figure(dpi=400)
plt.imshow(out1)
plt.axis("off")
plt.show()

mask = segments > 11
maskseg = segments.copy()
maskseg[~mask] = 0
plt.figure(dpi=400)
plt.imshow(segments)
plt.axis("off")
plt.show()
#%%========================ROW OVER TIME===========================
import matplotlib.pyplot as plt
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

pxstk = np.zeros((3712,int(size/3),3))
medimg = np.zeros((3712,3712,3))
k=1800
for j in range(int(size/3)):
    i=j*3
    redpx = plt.imread(mypath+sort[2,i])[:,k]
    greenpx = plt.imread(mypath+sort[2,i+1])[:,k]
    bluepx = plt.imread(mypath+sort[2,i+2])[:,k]
    pxstk[:,j,:]=np.array([redpx,greenpx,bluepx]).T
    #print(np.round(i*3/size*100))
medimg[:,k] = np.median(pxstk,axis=1)
print(())

#%%==========================CALCULATE STD======================================
import matplotlib.pyplot as plt
#error between 525 and 530 (x3)
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
meanimg = np.loadtxt(datapath+"meanimg.np").reshape((3712,3712,3)).copy()
imgstk = np.zeros((3712,3712,3))
for j in range(100):
    i = j*3
    diffr = plt.imread(mypath+sort[2,i])
    diffg = plt.imread(mypath+sort[2,i+1])
    diffb = plt.imread(mypath+sort[2,i+2])
    curdiff = np.array([diffr,diffg,diffb]).transpose(1,2,0)-meanimg

    imgstk=imgstk+curdiff
    if j%5==0:
        print(j)

imgvar = imgstk**2/(numimg)
#%%=========================PLOT STD===================================
std = np.sqrt(abs(imgvar))
meandiff = meanimg.copy()-std

plt.imshow(np.uint8(std))
print(np.max(meandiff))
#%%====================THRESHOLDING USING STD===================================
testimg = np.array([plt.imread(mypath+sort[2,0]),plt.imread(mypath+sort[2,1]),plt.imread(mypath+sort[2,2])]).transpose(1,2,0)

testthreshold = testimg-meanimg<std*0.3
thresholdstack = ~np.any(~testthreshold,axis=2)
cloudremoveall = testimg.copy()
cloudremoveone = testimg.copy()
cloudremoveall[~thresholdstack]=0
cloudremoveone[~testthreshold]=0
plt.figure(dpi=400)
plt.imshow(cloudremoveone)
plt.show()
plt.figure(dpi=400)
plt.imshow(cloudremoveall)
plt.show()      

#%%==========================SINGLE PIXEL CHANGES========================
%matplotlib qt
plt.figure(dpi=400)
x = np.linspace(0,int(2190/3),int(2190/3))
ones = np.ones(int(2190/3))
plt.plot(x,pxstk[0,:],color='red',alpha=0.6,linewidth=0.2)
plt.plot(x,pxstk[1,:],color='green',alpha=0.6,linewidth=0.2)
plt.plot(x,pxstk[2,:],color='blue',alpha=0.6,linewidth=0.2)
plt.plot(x,ones*np.median(pxstk[0,:]),color="red")
plt.plot(x,ones*np.median(pxstk[1,:]),color="green")
plt.plot(x,ones*np.median(pxstk[2,:]),color="blue")
plt.show()
#%%
transpimgplot = np.transpose(pxstk,(0,2,1))
plt.imshow(np.uint8(transpimgplot),aspect=780/3712)


