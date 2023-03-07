#K-means clustering implementation

# Imports ------------------------------------------
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




pixel_values = im_nmzd.reshape((-1, 3))

# Convert the data type to float32
pixel_values = np.float32(pixel_values)

# Define the parameters for k-means clustering
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
K = 3 # number of clusters
attempts = 10
flags = cv2.KMEANS_RANDOM_CENTERS

# Perform k-means clustering
retval, labels, centers = cv2.kmeans(pixel_values, K, None, criteria, attempts, flags)

# Convert the centers to integers
centers = np.uint8(centers)

# Reshape the labels back to the original image shape
labels = labels.flatten()
segmented_image = centers[labels.flatten()]

# Reshape the segmented image back to the original image shape
segmented_image = segmented_image.reshape(im_nmzd.shape)

# Display the segmented image
cv2.imshow('K-Means Segmentation', segmented_image)
cv2.waitKey(0)
cv2.destroyAllWindows()






