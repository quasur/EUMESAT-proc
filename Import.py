#%%
import numpy as np
import os

mypath = "C:/Users/adamc/Desktop/images"

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
    elif filenames[i][0:4] == "VIS6":
        props[0,i]=1
    elif filenames[i][0:4] == "VIS8":
        props[0,i]=2
    else:
        props[0,i]=3

    props[1,i] = int(filenames[i][29:43])#YYYY mm DD HH MM SS
    props[2,i] = filenames[i]                 #
                                             #
sort =props[:,props[1,:].argsort()]         #
#time ordered array of idexing: filter, datetime, filename 



