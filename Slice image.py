
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
#simple thresholding
fil_nm = sort[2, 2]

Vis6_1 =  plt.imread(os.path.join(mypath, fil_nm))
Vis6_1 = cv2.medianBlur(Vis6_1,5)

seg_1 = Vis6_1[0:500, :]
seg_2 = Vis6_1[500:1000, :]
seg_3 = Vis6_1[1000:1500, :]
seg_4 = Vis6_1[1500:3713, :]

Combined_1 = np.vstack((seg_1, seg_2))
Combined_2 = np.vstack((Combined_1, seg_3))
Combined_3 = np.vstack((Combined_2, seg_4))

ret, thr_seg_1 = cv2.threshold(seg_1, 80, 255, cv2.THRESH_BINARY)
ret2, thr_seg_2 = cv2.threshold(seg_2, 80, 255, cv2.THRESH_BINARY)
ret3, thr_seg_3 = cv2.threshold(seg_3, 80, 255, cv2.THRESH_BINARY)
ret4, thr_seg_4 = cv2.threshold(seg_4, 100, 255, cv2.THRESH_BINARY)

thr_Combined_1 = np.vstack((thr_seg_1, thr_seg_2))
thr_Combined_2 = np.vstack((thr_Combined_1, thr_seg_3))
thr_Combined_3 = np.vstack((thr_Combined_2, thr_seg_4))

fig6 = plt.figure()

ax5 = fig6.add_subplot(1, 2, 1)
ax6 = fig6.add_subplot(1, 2, 2)

ax5.imshow(Combined_3, cmap='Greys_r')
ax6.imshow(thr_Combined_3, cmap='Greys_r')

#%%
#adaptive thresholding
fil_nm = sort[2, 0]
    
Vis6_1 =  plt.imread(os.path.join(mypath, fil_nm))
Vis6_1 = cv2.medianBlur(Vis6_1,5)

th3 = cv2.adaptiveThreshold(Vis6_1, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,3,2)

fig5 = plt.figure()
ax1 = fig5.add_subplot(1, 2, 1)
ax2 = fig5.add_subplot(1, 2, 2)

ax1.imshow(Vis6_1, cmap='Greys_r')
ax2.imshow(th3, cmap='Greys')
