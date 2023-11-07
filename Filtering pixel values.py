
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
from scipy.ndimage import gaussian_filter1d
import scipy.signal

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


    

no_it = 0

def pixelwise_perpix(sort, mypath, indices, pixel):
    
    pixval_array = np.array([])

    no_it = 0
    
    for i in indices:
       

       fil_nm = sort[2, i]
           
       file_read = os.path.join(mypath, fil_nm[0])
           
       img = plt.imread(file_read)

       pixval = img[pixel, pixel]
       
       pixval_array = np.append(pixval_array, pixval)
       
       no_it+=1             
        
    print('Channel complete')    
    return pixval_array, no_it

#%%
pixelvalues_blue, no_it = pixelwise_perpix(sort, mypath, blue_indices, 1800)
pixelvalues_green, no_it = pixelwise_perpix(sort, mypath, green_indices, 1800)
pixelvalues_red, no_it = pixelwise_perpix(sort, mypath, red_indices, 1800)

x = np.linspace(0, len(pixelvalues_blue), len(pixelvalues_blue))


index = int(red_indices[12])
file_read = sort[2, index]
test_im = plt.imread(os.path.join(mypath, file_read))

#%%
fig10 = plt.figure()
ax1 = fig10.add_subplot(1, 1, 1)

y3 = scipy.signal.medfilt(pixelvalues, len(pixelvalues)+1)
ax1.plot(x, y3)
ax1.plot(x, pixelvalues)

#%%

#%%
