#Graph cut implementation
# Imports ------------------------------------------
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.future import graph


import histomicstk as htk

import cv2
import numpy as np
import scipy as sp

import skimage.io
import skimage.measure
import skimage.color

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
%matplotlib inline

plt.rcParams['figure.figsize'] = 10, 10
plt.rcParams['image.cmap'] = 'gray'
titlesize = 24

#Example image-----------------------------------------
input_image_file = ('https://data.kitware.com/api/v1/file/576ad39b8d777f1ecd6702f2/download')  

im_input = skimage.io.imread(input_image_file)[:, :, :3]

#Reference Img 1
ref_image_file = ('https://data.kitware.com/api/v1/file/57718cc28d777f1ecd8a883c/download')  

im_reference = skimage.io.imread(ref_image_file)[:, :, :3]

# Mean and STD calculation
mean_ref, std_ref = htk.preprocessing.color_conversion.lab_mean_std(im_reference)

# Reinhard color normalization
im_nmzd = htk.preprocessing.color_normalization.reinhard(im_input, mean_ref, std_ref)



segments = slic(img_nmzd, n_segments=300, compactness=10, sigma=1)

# Create a region adjacency graph
g = graph.rag_mean_color(img_nmzd, segments)

# Define the parameters for graph cut clustering
thresh = 50
mode = 'similarity'

# Perform graph cut clustering
labels = graph.cut_normalized(g, thresh, mode)

# Display the segmented image
fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(mark_boundaries(img_nmzd, labels))
ax.axis('off')
plt.show()
