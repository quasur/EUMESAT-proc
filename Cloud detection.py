#%%=====================IMPORT USED MODULES=============================
import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image, ImageFile
from skimage.filters import gaussian as blur
ImageFile.LOAD_TRUNCATED_IMAGES = True
#%%=====================CREATE STORTED FILES=============================
#MYPATH IS THE DIRECTORY FOR THE EUMETSAT RAW I,AGE
#DATAPATH IS THE DIRECTORY FOR THE MEAN AND STD FILES
#CHANGE THEM FOR THE CODE TO WORK
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
#%%==================SHOW COMPOUND IMAGE====================================

indx = 300
green = plt.imread(mypath+sort[2,indx+1])
red = plt.imread(mypath+sort[2,indx+0])
blue = plt.imread(mypath+sort[2,indx+2])


compound = np.array([red,green,blue]).transpose((1,2,0))
plt.figure(dpi=400)
plt.axis("off")
plt.title
plt.imshow(compound)

#%%========================PIXEL OVER TIME===========================
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
#initialise array
pxstk = np.zeros((numimg,3),dtype=np.uint8)
#import the rgb values of a single pixel throughout the images
for j in range(numimg):
    i=j*3
    redpx = plt.imread(mypath+sort[2,i])[1500,2000]
    greenpx = plt.imread(mypath+sort[2,i+1])[1500,2000]
    bluepx = plt.imread(mypath+sort[2,i+2])[1500,2000]
    pxstk[j]=np.array([redpx,greenpx,bluepx]).T

#plot the pixel changes
plt.plot(pxstk[:,0],'r-')
plt.plot(pxstk[:,1],'r-')
plt.plot(pxstk[:,2],'r-')
plt.show()

#%%Import mean and std values
meanimg = np.loadtxt(datapath+"meanimg.np").reshape((3712,3712,3))
stdbase = np.loadtxt(datapath+"std.np").reshape((3712,3712,3))*255
#%%#==========================GENERATE WEEK IMAGES==================================

#Blur the std using gaussian convolution
std = blur(stdbase,(3,3),mode="reflect")

#define threshold used for VIS channels
def threshold(inp,std):
    #crude mask for the red threshold
    #redthresh = np.logical_or(inp[:,:,0]<=-1*std[:,:,0],inp[:,:,0]>=2*std[:,:,1])#,inp[:,:,0]>=10*std[:,:,0])
    grethresh = np.logical_or(inp[:,:,1]>=0.5*std[:,:,1],inp[:,:,1]<=-2*std[:,:,1])##inp[:,:,1]<=-std[:,:,1])
    bluthresh = np.logical_or(inp[:,:,2]>=0.5*std[:,:,2],inp[:,:,0]<=-2*std[:,:,1])#inp[:,:,2]<=-std[:,:,2])
    gbthresh = np.any(np.array([grethresh,bluthresh]),axis=0)
    return gbthresh

#array initialisation 
weekstack = np.zeros((3712,3712,3))
redval = np.zeros((3712,3712,60),dtype=np.uint8)
weekweights = np.zeros((3712,3712),dtype=np.uint8)
months = 30*np.arange(23)+60
month=0

#first window initial average
for j in range(60):
    i = j*3
    redval[:,:,j]=plt.imread(mypath+sort[2,i]).astype(np.uint8)
    print(i)

#big loop that takes an hour to run
for j in range(numimg):
    i = (j+0)*3
    #import the cay image
    dayimg=np.array([plt.imread(mypath+sort[2,i]),
                    plt.imread(mypath+sort[2,i+1]),
                    plt.imread(mypath+sort[2,i+2])]).transpose(1,2,0)
    print(j)

    #==================Redpx moving avg===========================
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
    #========================Cloud detection and removal=========================
    #Cloud mask for red image
    reddiff = dayimg[:,:,0]-movingavg
    rmask = np.logical_or(reddiff<=-0.45*std[:,:,0],reddiff>=1*std[:,:,1])

    #Cloud mask for VIS channels
    gbmask = threshold(dayimg-meanimg,std)

    #cloud removal
    mask = np.logical_or(rmask,gbmask)
    dayimg[mask]=0
    weekstack=weekstack+dayimg

    #number of clouds in each pixel
    changes = np.ones((3712,3712))
    changes[mask]=0
    weekweights = weekweights + changes

    #weighted avg for cloud negation
    weekweightdiv = weekweights
    weekweightdiv[weekweights==0]=1
    weekimg = weekstack/np.repeat(weekweightdiv[:,:,np.newaxis],3,axis=2)

    #save the image every 30 days
    if np.any(j == months-1):
        print("saving data for month: ", month )
        wimg = weekimg-np.min(weekimg)
        img= np.uint8(255*wimg/np.max(wimg))
        fimname = str("monthimg/month"+str(month)+".png")
        imsave= Image.fromarray(img)
        imsave.save(fimname,format="png")
        month = month +1        
