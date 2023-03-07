#Chan-vese method implementation

# Imports ------------------------------------------
import histomicstk as htk
from skimage.segmentation import chan_vese

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


# Perform Chan-Vese clustering
cv = chan_vese(img_nmzd, mu=0.25, lambda1=1, lambda2=1, tol=1e-3, max_iter=200, dt=0.5)

# Display the segmented image
fig, ax = plt.subplots(figsize=(6, 6))
ax.imshow(img_nmzd, cmap='gray')
ax.contour(cv, [0.5], colors='r', linewidths=2)
ax.axis('off')
plt.show()
