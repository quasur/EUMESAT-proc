
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
    

pixval_red, no_it = pixelwise(sort, mypath, red_indices[60:120])
pixval_green, no_it = pixelwise(sort, mypath, green_indices[0:200])
pixval_blue, no_it = pixelwise(sort, mypath, blue_indices[0:200])

#%%
red_img = pixval_red/no_it
green_img = pixval_green/no_it
blue_img = pixval_blue/no_it

comb_image_1 = np.dstack((red_img, green_img)) 
comb_image_2 = np.dstack((comb_image_1, blue_img))

fig = plt.figure()
ax1 = fig.add_subplot(1, 1, 1)

ax1.imshow(np.uint8(red_img))

#%%
file_2_read = sort[2, 1]
test_image = plt.imread(os.path.join(mypath, file_2_read))
test_image_output = np.copy(test_image)


image_diff = np.abs(green_img-test_image)
bad_indices = np.argwhere(image_diff>13)

print(len(bad_indices))

for i in range(0, len(bad_indices)):
    
    ind_x = bad_indices[i, 0]
    ind_y = bad_indices[i, 1]
    
    good_pix = green_img[ind_x, ind_y]

    test_image_output[ind_x, ind_y] = good_pix

fig = plt.figure()
ax1 = fig.add_subplot(1, 2, 1)   
ax2 = fig.add_subplot(1, 2, 2)  
 
ax1.imshow(test_image)
ax2.imshow(test_image_output)   

#%%

avg_pix = green_img[1800, 1800]

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
    return pixval_array, no_it

pixel_array, no_it = pixelwise_singpix(sort, mypath, green_indices[0:200], 1800)

#%%
x = np.linspace(0, len(pixel_array), len(pixel_array))

plt.plot(x, pixel_array)
plt.plot(x, mean)