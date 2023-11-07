#%%=====================CREATE STORTED FILES=============================
import numpy as np
import os

mypath = "/Users/alex/Documents/Physics 4th Year/Imaging & date processing/Sat img/"
datapath="/Users/alex/Documents/Physics 4th Year/Imaging & date processing/"

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

meanimg = np.loadtxt(datapath+"meanimg.np").reshape((3712,3712,3)).copy()
image = meanimg[:, :, 0]

#%%
#Change the image array shape into q,3 rather than x,y,3
pxval=image.reshape((-1))
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

#reshape back into original dimention

segmentedimage = segmentedimage.reshape(image.shape)

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
#%%=============================SLIC=================================

#define constants
numSp =30 #number of superpixels
c=0.1      #colour weight
sigma =0 #factor for pre-processing gaussian blur
#calculate superpixel labels
segments =slic(meanimg,n_segments=numSp,compactness=c,sigma=sigma,enforce_connectivity=False,max_size_factor=3)
#post processing step changes all values in a superpixel to mean value
out1 = label2rgb(segments,image,kind="avg",bg_label=0)

#%%=========================SLIC PLOTTING===========================
#plot the images
plt.figure(dpi=400)
plt.imshow(np.uint8(out1))
plt.axis("off")
plt.show()

mask = segments > 11
maskseg = segments.copy()
maskseg[~mask] = 0

ret, threshold = cv2.threshold(out1, 87, 255, cv2.THRESH_BINARY)

plt.figure(dpi=400)
plt.imshow(np.uint8(threshold))
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

#%%=========================PLOT STD===================================

std = np.sqrt(abs(imgvar)/np.max(abs(imgvar)))*255
meandiff = meanimg.copy()-std
fig = plt.figure()


ax1 = fig.add_subplot(1, 1, 1)
ax1.imshow(np.uint8(std[:, :, 1]))

#%%====================THRESHOLDING USING STD===================================

flatstd = np.ndarray.flatten(std)
x = np.linspace(0, len(flatstd), len(flatstd))

fig3 = plt.figure()

ax10 = fig3.add_subplot(1, 1, 1)
ax10.plot(x, flatstd)






#%%
testimg = np.array([plt.imread(mypath+sort[2,0]),plt.imread(mypath+sort[2, 1]),plt.imread(mypath+sort[2,2])]).transpose(1,2,0)

testthreshold[:, :, 0] = abs((testimg[:, :, 0]-meanimg[:, :, 0])>std[:, :, 0])*1.2
testthreshold[:, :, 1] = (testimg[:, :, 1]-meanimg[:, :, 1])>std[:, :, 1]*.01
testthreshold[:, :, 2] = (testimg[:, :, 2]-meanimg[:, :, 2])>std[:, :, 2]*.01

thresholdstack = ~np.all(~testthreshold,axis=2)
cloudremoveall = testimg.copy()
cloudremoveone = testimg.copy()
cloudremoveall[thresholdstack]=0
cloudremoveone[testthreshold]=0

fig = plt.figure(dpi=400)

ax1 = fig.add_subplot(1, 2, 1)
ax2 = fig.add_subplot(1, 2, 2)
#ax3 = fig.add_subplot(1, 3, 3)

ax1.imshow(np.uint8(testimg))
ax1.set_xticks([])
ax1.set_yticks([])


ax2.imshow(cloudremoveall, cmap='Greys_r') 
ax2.set_xticks([])
ax2.set_yticks([])

'''  
ax3.imshow(np.uint8(std))
ax3.set_xticks([])
ax3.set_yticks([])
'''
#%
    

#%%    

'''

Plotting graphs for pixel variation

'''

from skimage.filters import gaussian as blur

std = blur(std, (3, 3), mode='reflect')

#%%
red_indices = np.argwhere(sort[0, :]==0)
green_indices = np.argwhere(sort[0, :]==1)
blue_indices = np.argwhere(sort[0, :]==2)

def pixelwise_singpix(sort, mypath, indices, pixel):
    
    pixval_array = np.array([])
    no_it = 0
    
    for i in indices:
        

        fil_nm = sort[2, i]
            
        file_read = os.path.join(mypath, fil_nm[0])
            
        img = plt.imread(file_read)
        
        pixval_array = np.append(pixval_array, img[pixel, pixel])
        
        no_it+=1
        
    print('Channel complete')    
    return pixval_array

#%%
pix_index = 1850

red_pix_val = pixelwise_singpix(sort, mypath, red_indices, pix_index)
vis8_pix_val = pixelwise_singpix(sort, mypath, green_indices, pix_index)
vis6_pix_val = pixelwise_singpix(sort, mypath, blue_indices, pix_index)

#%%
std_red_pixel = std[pix_index, pix_index, 0]
std_VIS8_pixel = std[pix_index, pix_index, 1]
std_VIS6_pixel = std[pix_index, pix_index, 2]

#%%
x = np.linspace(0, len(red_pix_val), len(red_pix_val))

#mean pixel value
mean_red_pixel = meanimg[pix_index, pix_index, 0]
mean_green_pixel = meanimg[pix_index, pix_index, 1]
mean_blue_pixel = meanimg[pix_index, pix_index, 2]


y = np.ones_like(red_pix_val)
#%%
'''
IR plot
'''
fig = plt.figure(figsize=(12, 8))
ax1=plt.axes()


ax1.plot(x/2, red_pix_val-mean_red_pixel, 'r', label='Difference from the mean value')
ax1.set_title('Temporal variation of IR16 pixel [1600, 1600]')
ax1.set_xlabel('Day of image acquisition')
ax1.set_ylabel('Pixel brightness')

#%%
'''
VIS8 plot
'''
fig = plt.figure(figsize=(12, 8))
ax1=plt.axes()


ax1.plot(x/2, vis8_pix_val-mean_green_pixel, 'g', label='Difference from the mean value')
ax1.set_title('Temporal variation of VIS8 pixel [1600, 1600]')
ax1.set_xlabel('Day of image acquisition')
ax1.set_ylabel('Pixel brightness')


#%%
'''
VIS6 plot
'''
fig = plt.figure(figsize=(12, 8))
ax1=plt.axes()


ax1.plot(x/2, vis6_pix_val-mean_blue_pixel, 'b', label='Difference from the mean value')
ax1.set_title('Temporal variation of VIS6 pixel [1600, 1600]')
ax1.set_xlabel('Day of image acquisition')
ax1.set_ylabel('Pixel brightness')



