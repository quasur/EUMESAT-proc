
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

#%%

def slic_image(image, mask):
    
    #mask the image
    test = mask*image
    
    #define constants
    numSp = 35 #number of superpixels
    c= 1  #colour weight
    sigma = .5 #factor for pre-processing gaussian blur

    #calculate superpixel labels
    segments =slic(test, n_segments=numSp,compactness=c,sigma=sigma,enforce_connectivity=False,max_size_factor=3,channel_axis=-1)
    #post processing step changes all values in a superpixel to mean value
    out1 = np.uint8(label2rgb(segments, test, kind="avg",bg_label=0))
    
    return out1

def kmeans_image(image):
    
    #thresholding based on red channel value, might have to tweak
    slic_mask = cv2.inRange(image[:, :, 0],0, 85)
    mask_ind = np.argwhere(slic_mask==255)
    mask_out = np.copy(image)
    
    #sets water indices to zero
    for i in range(0, len(mask_ind)):
        
        ind_x = mask_ind[i,0]
        ind_y = mask_ind[i,1]
        
        mask_out[ind_x, ind_y] = 0
        
    #defines target image    
    test = np.uint8(mask_out)
    """
    #Change the image array shape into q,3 rather than x,y,3
    pxval=test.reshape((-1))
    pxval =np.float32(pxval)

    #number of means 
    k= 4

    #Error threshold
    E_Val = 0.0001
    #Max iterations                        
    max_iterations = 10000


    initial_means = cv2.KMEANS_PP_CENTERS

    #Define the stopping criteria as a single variable to input into the function
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,max_iterations,E_Val)
    #calculate the k-means clusters
    compactness,labels,(centers)=cv2.kmeans(pxval, k,None,criteria,5,initial_means)

    #data formatting required for functions to play nice
    centers = np.uint8(centers)
    labels=labels.flatten()

    #convert pixels to colour of centroid

    segmentedimage = centers[labels.flatten()]

    #reshape back into original dimension

    segmentedimage = segmentedimage.reshape(meanimg.shape)
    """
    return mask_out#segmentedimage

#%%===============BADPX INTERPOLATION==============================
datapath="C:/Users/adamc/Desktop/bigdata/"
meanimg =np.uint8(np.loadtxt(datapath+"meanimg.np").reshape((3712,3712,3)))
#%%
mask = plt.imread("mask.png")[:,:,0:3]

biome = np.zeros((3712,3712,23),dtype=np.uint8)
ones = np.ones((3712,3712),dtype=np.uint8)

for i in range(23):
    print(i)
    img = np.uint8(plt.imread("monthimg/month"+str(i)+".png")*255)
    fiximg = img.copy()
    #set "missing pixels" to the corresponding value in the mean img
    badpx = np.all(fiximg==0,axis=2)
    fiximg[badpx]=meanimg[badpx]


    out1 =slic_image(fiximg,mask)
    groupimg= kmeans_image(out1)

    curbiome =np.zeros((3712,3712),dtype=np.uint8)
    desertmask1 = np.any(groupimg[:,:,:]>=1,axis=2)
    forestmask =np.logical_and((groupimg[:,:,1]+15 )>(groupimg[:,:,0]),desertmask1)
    aridmask =  np.logical_and((groupimg[:,:,1]+15)<(groupimg[:,:,0]),desertmask1)
    desertmask2 = groupimg[:,:,2]>100
    curbiome[desertmask1]=1
    curbiome[aridmask]=1
    curbiome[forestmask]=2
    curbiome[desertmask2]=1
    biome[:,:,i]=curbiome

#%%
biomeones = np.ones((3712,3712,23))
biomeones[biome!=0]=0
weights = np.sum(biomeones,axis=1)
biomerows = np.sum(biome,axis=1)/weights
biomerowsavg = np.repeat(np.mean(biomerows,axis=1)[:,np.newaxis],23,axis=1)
biomerowsnorm = biomerows-biomerowsavg
plt.imshow(biomerowsnorm,aspect=23/3712)




#%%


from mpl_toolkits.mplot3d import Axes3D
from matplotlib import colors

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, projection="3d")

for i in range(6):
    print(i)
    img = np.uint8(plt.imread("monthimg/month"+str(i)+".png")*255)
    fiximg = img.copy()
    badpx = np.all(fiximg==0,axis=2)
    fiximg[badpx]=meanimg[badpx]



    out1 = kmeans_image(slic_image(fiximg,mask))
    #testimg2= kmeans_image(testimg)
    #plt.imshow(testimg2)


    #removing duplicates


    spix_val = np.unique(out1.reshape(-1, out1.shape[2]), axis=0)

    # Extract RGB components
    r = spix_val[:, 0]
    g = spix_val[:, 1]
    b = spix_val[:, 2]

    col_val = []

    for i in range(np.size(spix_val[:,0])):
        
        pixel_colors = spix_val[i,:]
        pixel_colors = pixel_colors/255
        
        print(pixel_colors)

        pixel_colors = pixel_colors.tolist()
        
        col_val.append(pixel_colors)

    ax.scatter(r, g, b, '.' ,facecolors=col_val, s=50)
    
# Set axis labels
ax.set_xlabel('Red')
ax.set_ylabel('Green')
ax.set_zlabel('Blue')

#Show the plot
plt.show()

plt.figure(dpi=400)
plt.imshow(np.uint8(out1), cmap='Greys_r')
plt.axis("off")
plt.title('Thresholded slic image')
plt.show()



