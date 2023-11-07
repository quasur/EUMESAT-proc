
#%%
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from PIL import Image

mypath = "/Users/alex/Documents/Physics 4th Year/Imaging & date processing/Sat img"

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

    props[1,i] = int(filenames[i][29:43]) #YYYY mm DD HH MM SS
    props[2,i] = filenames[i]

    
sort = props[:, np.lexsort((props[0,:], props[1, :]))]

#time ordered array of idexing: filter, datetime, filename 

#%%
import glob 
import numpy.ma as ma

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

image_arr = imagearray(filenames)
#%%   

(T, threshInv) = cv2.threshold(IR, 200, 255,
	cv2.THRESH_BINARY_INV)
