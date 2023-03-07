#Otsu_thresholding method
import cv2
import histomicstk as htk

import numpy as np
import scipy as sp

import skimage.io
import skimage.measure
import skimage.color

#Example image-----------------------------------------
input_image_file = ('https://data.kitware.com/api/v1/file/576ad39b8d777f1ecd6702f2/download')  

im_input = skimage.io.imread(input_image_file)[:, :, :3]

#plt.imshow(im_input)
#_ = plt.title('Input Image', fontsize=16)


ret, thresh = cv2.threshold(im_input, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Display the thresholded image
cv2.imshow('Otsu Threshold', thresh)
cv2.waitKey(0)
cv2.destroyAllWindows()
