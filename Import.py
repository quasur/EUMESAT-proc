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
for j in range(numimg):
    i = j*3
    diffr = plt.imread(mypath+sort[2,i])
    diffg = plt.imread(mypath+sort[2,i+1])
    diffb = plt.imread(mypath+sort[2,i+2])
    curdiff = np.array([diffr,diffg,diffb]).transpose(1,2,0)-meanimg

    imgstk=imgstk+curdiff
    if j%5==0:
        print(j)

imgvar = imgstk**2/(numimg)
std = np.sqrt(abs(imgvar))
#%%
import matplotlib.pyplot as plt

std = np.loadtxt(datapath+"std.np").reshape((3712,3712,3))
print(np.max(std))
plt.imshow(np.uint8(std*255))

#%%====================THRESHOLDING USING STD===================================
meanimg = np.loadtxt(datapath+"meanimg.np").reshape((3712,3712,3))
stdbase = np.loadtxt(datapath+"std.np").reshape((3712,3712,3))*255
#%%#==========================GENERATE WEEK IMAGES==================================
import matplotlib.pyplot as plt
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


from skimage.filters import gaussian as blur
std = blur(stdbase,(3,3),mode="reflect")

testimg = np.array([plt.imread(mypath+sort[2,0]),plt.imread(mypath+sort[2,1]),plt.imread(mypath+sort[2,2])]).transpose(1,2,0)
diffimg = testimg-meanimg
def threshold(inp,std):
    #crude mask for the red threshold
    #redthresh = np.logical_or(inp[:,:,0]<=-1*std[:,:,0],inp[:,:,0]>=2*std[:,:,1])#,inp[:,:,0]>=10*std[:,:,0])
    grethresh = np.logical_or(inp[:,:,1]>=0.5*std[:,:,1],inp[:,:,1]<=-2*std[:,:,1])##inp[:,:,1]<=-std[:,:,1])
    bluthresh = np.logical_or(inp[:,:,2]>=0.5*std[:,:,2],inp[:,:,0]<=-2*std[:,:,1])#inp[:,:,2]<=-std[:,:,2])
    gbthresh = np.any(np.array([grethresh,bluthresh]),axis=0)
    return gbthresh


#testthresh = threshold(diffimg,std)
#plt.imshow(np.uint8(std))
#plt.show()
#testimg[testthresh]=0
#plt.figure()
#plt.imshow(testimg[:,:,:])


#meanimg = np.loadtxt(datapath+"meanimg.np").reshape((3712,3712,3)).copy()
#std = np.loadtxt(datapath+"std.np").reshape((3712,3712,3)).copy()*255

weeks=numimg/10
weekstack = np.zeros((3712,3712,3))
redval = np.zeros((3712,3712,60),dtype=np.uint8)
weekweights = np.zeros((3712,3712),dtype=np.uint8)

#first window initial average
for j in range(60):
    i = j*3
    redval[:,:,j]=plt.imread(mypath+sort[2,i]).astype(np.uint8)
    print(i)

for j in range(60):
    i = (j+180)*3
    dayimg=np.array([plt.imread(mypath+sort[2,i]),
                    plt.imread(mypath+sort[2,i+1]),
                    plt.imread(mypath+sort[2,i+2])]).transpose(1,2,0)
    print(j)


    #the first window repeats for the first 30 avgs
    if j <30:
        movingavg=np.sum(redval,axis=2)/60
        print(30)

    #middle windows
    #load in one image at a time and shuttle the others off one by one until the last image is loaded
    elif j >=30 & j<700:
        movingavg=np.sum(redval,axis=2)/60
        redval[:,:,:-1]=redval[:,:,1:]
        redval[:,:,-1]=plt.imread(mypath+sort[2,i]).astype(np.uint8)
        print(700)

    #pad the other edge for 30 values
    elif j >= 700:
        movingavg=np.sum(redval,axis=2)/60
        #for i in range(30):
        print(730)

    reddiff = dayimg[:,:,0]-movingavg
    rmask = np.logical_or(reddiff<=-2*std[:,:,0],reddiff>=2*std[:,:,1])

    #change this to be the threshold funciton
    gbmask = threshold(dayimg-meanimg,std)

    mask = np.logical_or(rmask,gbmask)
    dayimg[mask]=0
    weekstack=weekstack+dayimg

    changes = np.ones((3712,3712))
    changes[mask]=0
    weekweights = weekweights + changes

badpx = np.argwhere(weekweights==1)
weekweightdiv = weekweights
weekweightdiv[weekweights==0]=1
weekimg = weekstack/np.repeat(weekweightdiv[:,:,np.newaxis],3,axis=2)
#%%
%matplotlib inline
plt.figure(dpi=400)
wimg = weekimg-np.min(weekimg)
img= np.uint8(255*wimg/np.max(wimg))
plt.axis("off")
plt.imshow(img)
from PIL import Image
imsave= Image.fromarray(img)
imsave.save("month3.png",format="png")
# %% ================Adapting code to threshold red=============
import matplotlib.pyplot as plt
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
redval = np.zeros((3712,3712,60),dtype=np.uint8)
#first window initial average
if i ==0:
    indx = i*3
    redval[:,:,i]=plt.imread(mypath+sort[2,indx]).astype(np.uint8)
    print(0)
#the first window repeats for the first 30 avgs
if i <30:
    movingavg=np.sum(redval,axis=2)/60
    print(30)

#middle windows
#load in one image at a time and shuttle the others off one by one until the last image is loaded
if i >=30 & i<700:
    #for i in range(numimg-60):
    indx=(i+60)*3
    movingavg=np.sum(redval,axis=2)/60
    redval[:,:,:-1]=redval[:,:,1:]
    redval[:,:,-1]=plt.imread(mypath+sort[2,indx]).astype(np.uint8)
    pxval[i+30]=movingavg[1800,1800]
    print(700)

#pad the other edges for 30 values
if i >= 700:
    movingavg=np.sum(redval,axis=2)/60
    #for i in range(30):
    pxval[-30+i]=movingavg[1800,1800]
    print(730)

#%%==================2D COLOUR HISTOGRAM===============================
import matplotlib.pyplot as plt
r=plt.imread(mypath+sort[2,0]).ravel()
g=plt.imread(mypath+sort[2,1]).ravel()
b=plt.imread(mypath+sort[2,2]).ravel()

rg,xe1,ye1 =np.histogram2d(np.asarray(r),np.asarray(g),bins=173)
rb,xe2,ye2 =np.histogram2d(np.asarray(r),np.asarray(b),bins=173)
gb,xe3,ye3 =np.histogram2d(np.asarray(g),np.asarray(b),bins=173)

plt.imshow(np.log(rg+1),origin="lower")
plt.show()
plt.figure()
plt.imshow(np.log(rb+1),origin="lower")
plt.show()
plt.figure()
plt.imshow(np.log(gb+1),origin="lower")
plt.show()

#%%========================COLOUR SPACE PLOTS===========================
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
pxval = np.zeros((10000,numimg,3))
for i in range(numimg):
    indx = i*3
    pxval[:,i,0]=plt.imread(mypath+sort[2,indx])[1750:1850,1750:1850].ravel()
    pxval[:,i,1]=plt.imread(mypath+sort[2,indx+1])[1750:1850,1750:1850].ravel()
    pxval[:,i,2]=plt.imread(mypath+sort[2,indx+2])[1750:1850,1750:1850].ravel()
#%%

r=pxval[:,:,0].ravel()
g=pxval[:,:,1].ravel()
b=pxval[:,:,2].ravel()

rg,xe1,ye1 =np.histogram2d(np.asarray(r),np.asarray(g),bins=173)
rb,xe2,ye2 =np.histogram2d(np.asarray(r),np.asarray(b),bins=173)
gb,xe3,ye3 =np.histogram2d(np.asarray(g),np.asarray(b),bins=173)

plt.figure(dpi=400)
plt.imshow(np.log(rg+1),origin="lower")
plt.xlabel("r")
plt.ylabel("g")
plt.show()
plt.figure(dpi=400)
plt.imshow(np.log(rb+1),origin="lower")
plt.xlabel("r")
plt.ylabel("b")
plt.show()
plt.figure(dpi=400)
plt.imshow(np.log(gb+1),origin="lower")
plt.xlabel("g")
plt.ylabel("b")
plt.show()
#%% ==========================Moving average for red band====================================
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
redval = np.zeros((numimg),dtype=np.uint8)
for i in range(numimg):
    indx = i*3
    redval[i]=plt.imread(mypath+sort[2,indx])[1500,1500].astype(np.uint8)


movingavgpx = np.zeros(numimg-60)
for i in range(numimg-60):
    movingavgpx[i]=np.sum(redval[i:i+60])/60
movingavgpx = np.pad(movingavgpx,30,mode="edge")
#%%
plt.plot(redval,'b-')
plt.plot(movingavgpx,'r-')
std=np.std(redval)
plt.plot(movingavgpx+1.5*std,'g')
plt.plot(movingavgpx-1*std,'g')

#%%
%matplotlib qt
plt.plot(pxval)
plt.plot(movingavgpx,'r')

#%%
