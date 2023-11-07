
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

#%%===============BADPX INTERPOLATION==============================

img = plt.imread("monthimg/month0.png")





#%%=============================SLIC=================================
#define constants
numSp =25 #number of superpixels
c=1      #colour weight
sigma =3 #factor for pre-processing gaussian blur
image = img
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
#%%
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

