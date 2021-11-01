"""
References
- https://learnopencv.com/seamless-cloning-using-opencv-python-cpp/
- https://docs.opencv.org/4.5.4/df/da0/group__photo__clone.html
"""

# import cv2
from cv2 import cv2
import numpy as np

src = cv2.imread("images/src_img.jpg")
dst = cv2.imread("images/dest_img.jpg")

src_mask = cv2.imread("images/src_img_rough_mask.jpg")
#src_mask = cv2.imread("images/src_img_rough_mask2.jpg")
src_mask = np.invert(src_mask)

# Create a rough mask around the airplane.
# src_mask = np.zeros(src.shape, src.dtype)
# poly = np.array([ [4,80], [30,54], [151,63], [254,37], [298,90], [272,134], [43,122] ], np.int32)
# cv2.fillPoly(src_mask, [poly], (255, 255, 255))

cv2.namedWindow('src_mask', cv2.WINDOW_NORMAL)
cv2.imshow('src_mask', src_mask)
cv2.waitKey(0)

# Where to place image.
center = (500,500)

# Clone seamlessly.
output = cv2.seamlessClone(src, dst, src_mask, center, cv2.NORMAL_CLONE)
# output = cv2.seamlessClone(src, dst, src_mask, center, cv2.MIXED_CLONE)

# Write result
cv2.imwrite("images/opencv-seamless-cloning-example.jpg", output)

cv2.namedWindow('output', cv2.WINDOW_NORMAL)
cv2.imshow('output', output)
cv2.waitKey(0)