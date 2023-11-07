
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
import glob 
import numpy.ma as ma
from PIL import Image, ImageFile 

ImageFile.LOAD_TRUNCATED_IMAGES = True

filenames = sort[2, 6:12]

properties = sort[:, 6:12]

def imagearray(filenames):
    
    images = []
    
    for i in range(0, len(filenames)):
    
        file_read = str(filenames[i])
    
        for name in glob.glob(os.path.join(mypath, file_read)):
    
            img = plt.imread(name)
    
            if img is not None:
                
                images.append(img)       
    
    image_arr = np.array(images)
        
    return image_arr

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
    

pixval_red, no_it = pixelwise(sort, mypath, red_indices[0:30])
red_img = pixval_red/no_it

#%%
pixval_green, no_it = pixelwise(sort, mypath, green_indices) 
pixval_blue, no_it = pixelwise(sort, mypath, blue_indices)

blue_img = pixval_blue/no_it
green_img = pixval_green/no_it


#%%
fig5 = plt.figure()
ax1 = fig5.add_subplot(1, 1, 1)

ax1.imshow(red_img, cmap='Greys_r')

#%%

file_2_read = sort[2, 0]
test_image = plt.imread(os.path.join(mypath, file_2_read))
test_image_output = np.copy(test_image)

fig = plt.figure()

ax1 = fig.add_subplot(1, 3, 1)   
ax2 = fig.add_subplot(1, 3, 2)
ax3 = fig.add_subplot(1, 3, 3)

#%%

image_diff = np.abs(np.subtract(red_img, test_image))
cloud_indices = np.argwhere(image_diff>10)
mask_array =  np.zeros_like(test_image)

for j in range(0, len(cloud_indices)):
    
    ind_x = cloud_indices[j, 0]
    ind_y = cloud_indices[j, 1]
    
    mask_array[ind_x, ind_y] = 1
    
#%%
pix_op = 0

for i in range(0, len(cloud_indices)):
    
    ind_x = cloud_indices[i, 0]
    ind_y = cloud_indices[i, 1]
    
    good_pix = blue_img[ind_x, ind_y]    
    test_image_output[ind_x, ind_y] = good_pix

fig = plt.figure()

ax1 = fig.add_subplot(1, 3, 1)   
ax2 = fig.add_subplot(1, 3, 2)
ax3 = fig.add_subplot(1, 3, 3)
 
ax1.imshow(test_image, cmap='Greys_r')
ax2.imshow(test_image_output, cmap='Greys_r') 
ax3.imshow(mask_array, cmap='Greys_r')  

