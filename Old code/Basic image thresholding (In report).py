
#%% Produces plots for the basic thesholds that we tested early in the project
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt


mypath = #mypath is the location of all of the dataset files within the device

from os import walk

#produces 
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

#%%
red_file = sort[2, 0]
red_image = plt.imread(os.path.join(mypath, red_file))

VIS8_file = sort[2, 1]
VIS8_image = plt.imread(os.path.join(mypath, VIS8_file))

VIS6_file = sort[2, 2]
VIS6_image = plt.imread(os.path.join(mypath, VIS6_file))

ret, red_threshold = cv2.threshold(red_image,100, 255, cv2.THRESH_BINARY)
ret, VIS8_threshold = cv2.threshold(VIS8_image, 100, 255, cv2.THRESH_BINARY)
ret, VIS6_threshold = cv2.threshold(VIS6_image, 100, 255, cv2.THRESH_BINARY)

fig = plt.figure()
plt.subplots_adjust(wspace=0.05, hspace=0.25)

ax1 = fig.add_subplot(3, 3, 1)
ax2 = fig.add_subplot(3, 3, 2)
ax3 = fig.add_subplot(3, 3, 3)
ax4 = fig.add_subplot(3, 3, 4)   
ax5 = fig.add_subplot(3, 3, 5)  
ax6 = fig.add_subplot(3, 3, 6)  

ax1.set_xticks([])
ax2.set_xticks([])
ax3.set_xticks([])
ax4.set_xticks([])
ax5.set_xticks([])
ax6.set_xticks([])

ax1.set_yticks([])
ax2.set_yticks([])
ax3.set_yticks([])
ax4.set_yticks([])
ax5.set_yticks([])
ax6.set_yticks([])

ax1.set_title('NIR image')
ax2.set_title('VIS8 image')
ax3.set_title('VIS6 image')
ax4.set_title('NIR cloud mask')
ax5.set_title('VIS8 cloud mask')
ax6.set_title('VIS6 cloud mask')

ax1.imshow(red_image, cmap='Greys_r')  
ax2.imshow(VIS8_image, cmap='Greys_r')  
ax3.imshow(VIS6_image, cmap='Greys_r')  
ax4.imshow(red_threshold,  cmap='binary')
ax5.imshow(VIS8_threshold,  cmap='binary')  
ax6.imshow(VIS6_threshold, cmap='binary') 


#%%