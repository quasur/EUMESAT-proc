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

indx = 300
green = plt.imread(mypath+sort[2,indx+1])
red = plt.imread(mypath+sort[2,indx+0])
blue = plt.imread(mypath+sort[2,indx+2])


compound = np.array([red,green,blue]).transpose((1,2,0))
plt.figure(dpi=400)
plt.axis("off")
plt.title
plt.imshow(compound)
#%%
fig,ax = plt.subplots(2,2,dpi=300,figsize=(8,8))
#plt.subplots_adjust(wspace=-0.01,hspace=.1)
fig.tight_layout()
ax[0,0].imshow(red,cmap="gray")
ax[0,0].axis("off")
ax[0,0].set_title("IR1.6")
ax[0,1].imshow(green,cmap="gray")
ax[0,1].set_title("VIS0.8")
ax[0,1].axis("off")
ax[1,0].imshow(blue,cmap="gray")
ax[1,0].set_title("VIS0.6")
ax[1,0].axis("off")
ax[1,1].imshow(compound)
ax[1,1].axis("off")
ax[1,1].set_title("Compound image")
fig.savefig("RepIm/Base Images.png")
plt.show()
print(type(compound[1500,1500,1]))

#%%========================PIXEL OVER TIME===========================
import matplotlib.pyplot as plt
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

pxstk = np.zeros((numimg,3),dtype=np.uint8)
k=1800
for j in range(numimg):
    i=j*3
    redpx = plt.imread(mypath+sort[2,i])[1500,2000]
    greenpx = plt.imread(mypath+sort[2,i+1])[1500,2000]
    bluepx = plt.imread(mypath+sort[2,i+2])[1500,2000]
    pxstk[j]=np.array([redpx,greenpx,bluepx]).T
    #print(np.round(i*3/size*100))
#%%
%matplotlib inline
fig,ax = plt.subplots(3,dpi=300,figsize=(15,8))
plt.subplots_adjust(wspace=-0.01,hspace=0,top=0.95,bottom=0.1)
fig.suptitle("Pixel (1500,2000) colour vs time")
ax[0].plot(pxstk[:,0],'r-')
ax[2].set_xlabel("Image number")
ax[0].set_ylabel("IR1.6 Intensity")
ax[1].set_ylabel("VIS0.8 Intensity")
ax[2].set_ylabel("VIS0.6 Intensity")

ax[0].xaxis.set_tick_params(labelbottom=False)
ax[1].xaxis.set_tick_params(labelbottom=False)

ax[1].plot(pxstk[:,1],'g-')
ax[2].plot(pxstk[:,2],'b-')
fig.savefig("RepIm/PixelTime.png")

#%%==========================CALCULATE STD======================================
import matplotlib.pyplot as plt
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
months = 30*np.arange(23)+60
month=0

#first window initial average
for j in range(60):
    i = j*3
    redval[:,:,j]=plt.imread(mypath+sort[2,i]).astype(np.uint8)
    print(i)

for j in range(numimg):
    i = (j+0)*3
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
    rmask = np.logical_or(reddiff<=-0.45*std[:,:,0],reddiff>=1*std[:,:,1])

    #change this to be the threshold funciton
    gbmask = threshold(dayimg-meanimg,std)

    mask = np.logical_or(rmask,gbmask)
    dayimg[mask]=0
    weekstack=weekstack+dayimg

    changes = np.ones((3712,3712))
    changes[mask]=0
    weekweights = weekweights + changes


    #badpx = np.argwhere(weekweights==0)
    weekweightdiv = weekweights
    weekweightdiv[weekweights==0]=1
    weekimg = weekstack/np.repeat(weekweightdiv[:,:,np.newaxis],3,axis=2)

    if np.any(j == months-1):
        print("saving data for month: ", month )
        wimg = weekimg-np.min(weekimg)
        img= np.uint8(255*wimg/np.max(wimg))
        fimname = str("monthimg/month"+str(month)+".png")
        imsave= Image.fromarray(img)
        imsave.save(fimname,format="png")
        month = month +1        

#%%
%matplotlib qt
plt.figure(dpi=400)
plt.axis("off")
plt.imshow(img)
from PIL import Image

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
testimg = np.array([plt.imread(mypath+sort[2,0]),plt.imread(mypath+sort[2,1]),plt.imread(mypath+sort[2,2])]).transpose(1,2,0)
#first window initial average
for j in range(60):
    i = j*3
    redval[:,:,j]=plt.imread(mypath+sort[2,i]).astype(np.uint8)
    print(i)

movingavg=np.sum(redval,axis=2)/60
reddiff = testimg[:,:,0]-movingavg
rmask = np.logical_or(reddiff<=-0.45*std[:,:,0],reddiff>=1*std[:,:,1])
implot = testimg
implot[rmask]=0

%matplotlib qt
plt.figure(dpi=400)
plt.imshow(implot,cmap="gray")
#%%
meanimg = (np.loadtxt(datapath+"meanimg.np").reshape((3712,3712,3)).copy()).astype(np.uint8)
std= (np.loadtxt(datapath+"std.np").reshape((3712,3712,3))*255).astype(np.uint8)
#%%
fig,ax = plt.subplots(1,2,dpi=300,figsize=(6,3))
plt.subplots_adjust(wspace=0.01,hspace=0.05,top=0.92,bottom=0,left=0,right=1)
ax[0].axis("off")
ax[1].axis("off")
ax[0].set_title("mean pixel colours")
ax[1].set_title("standard deviaiton of each pixel")
ax[0].imshow(meanimg)
ax[1].imshow(std)
fig.savefig("RepIm/mean std.png")
# %% ================Adapting code to threshold red=============
import matplotlib.pyplot as plt
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
redval = np.zeros((60))
#first window initial average

for j in range(60):
    i = j*3
    redval[j]=plt.imread(mypath+sort[2,i])[1500,2000]
    print(i)

pxstk = np.zeros((numimg,3))
redavg = np.zeros(numimg)

for i in range(numimg):
    indx=i*3
    print(i,indx)
    #the first window repeats for the first 30 avgs
    if i <30:
        movingavg=np.sum(redval)/60
        redavg[i]=movingavg
        print(30)

    #middle windows
    #load in one image at a time and shuttle the others off one by one until the last image is loaded
    elif i >=30 & i<700:
        #for i in range(numimg-60):
        movingavg=np.sum(redval)/60
        redval[:-1]=redval[1:]
        redval[-1]=plt.imread(mypath+sort[2,indx])[1500,2000]
        redavg[i]=movingavg
        print(700)

    #pad the other edges for 30 values
    elif i >= 700:
        movingavg=np.sum(redval)/60
        #for i in range(30):
        redavg[-30+i]=movingavg
        print(730)

    redpx = plt.imread(mypath+sort[2,indx])[1500,2000]
    greenpx = plt.imread(mypath+sort[2,indx+1])[1500,2000]
    bluepx = plt.imread(mypath+sort[2,indx+2])[1500,2000]
    pxstk[i]=np.array([redpx,greenpx,bluepx]).T


#%%
meanimg = (np.loadtxt(datapath+"meanimg.np").reshape((3712,3712,3)).copy())[1500,2000]
std0= (np.loadtxt(datapath+"std.np")).reshape((3712,3712,3))
#%%
x=np.linspace(0,729,730)
%matplotlib inline
from skimage.filters import gaussian as blur
std1 = blur(std0,(3,3),mode="reflect")
mean = np.ones((730,3))*meanimg
std = np.ones((730,3))*std1[1500,2000]

fig,ax = plt.subplots(3,dpi=400,figsize=(9,4.5))
plt.subplots_adjust(wspace=-0.01,hspace=0,top=0.94,bottom=0.1,left=0.1,right=0.95)
fig.suptitle("Pixel (1500,2000) colour vs time with thresholds")
ax[2].set_xlabel("Image number")
ax[0].set_ylabel("IR1.6 Intensity")
ax[1].set_ylabel("VIS0.8 Intensity")
ax[2].set_ylabel("VIS0.6 Intensity")

ax[0].xaxis.set_tick_params(labelbottom=False)
ax[1].xaxis.set_tick_params(labelbottom=False)

ax[0].plot(pxstk[:,0],'r-')
ax[1].plot(pxstk[:,1],'g-')
ax[2].plot(pxstk[:,2],'b-')

ax[0].plot(redavg,color="magenta")
ax[1].plot(x,mean[:,1],color="magenta")
ax[2].plot(x,mean[:,2],color="magenta")

ax[0].plot(redavg-0.45*255*std[:,0],'k-')
ax[1].plot(mean[:,1]-255*std[:,1],'k-')
ax[2].plot(mean[:,2]-255*std[:,2],'k-')

ax[0].plot(redavg+255*std[:,0],'k-')
ax[1].plot(mean[:,1]+0.5*255*std[:,1],'k-')
ax[2].plot(mean[:,2]+0.5*255*std[:,2],'k-')

fig.savefig("RepIm/PixelTimeThresh.png")

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
months = 30*np.arange(23)+60
month=0

#first window initial average
for j in range(60):
    i = j*3
    redval[:,:,j]=plt.imread(mypath+sort[2,i]).astype(np.uint8)
    print(i)

for k in range(60):
    j=k+240
    i = (j)*3
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
    rmask = np.logical_or(reddiff<=-0.45*std[:,:,0],reddiff>=1*std[:,:,1])

    #change this to be the threshold funciton
    gbmask = threshold(dayimg-meanimg,std)

    mask = np.logical_or(rmask,gbmask)
    dayimg[mask]=0
#%%
plt.figure(dpi=400)
plt.imshow(dayimg)
plt.title("Cloud removed image")
plt.axis("off")
plt.savefig("RepIm/CloudZero.png")
    