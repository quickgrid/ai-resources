"""AKAZE, KAZE, SIFT keypoint matching with saving descriptors in hdf5, npy format.

Notes
    - Saving and loading with h5py seems faster than numpy save.
References
    - https://stackoverflow.com/a/65864622/1689698.
"""

import time
import cv2
import matplotlib.pyplot as plt
import numpy as np
import h5py

overall_start_time = time.time()

# Open and convert the input and training-set image from BGR to GRAYSCALE
image1 = cv2.imread(filename ='imgs/1.png',
                    flags = cv2.IMREAD_GRAYSCALE)

image2 = cv2.imread(filename ='imgs/8.png',
                    flags = cv2.IMREAD_GRAYSCALE)

# Initiate keypoint and descriptor extraction algorithm
keypoint_algorithm = cv2.AKAZE_create()
#keypoint_algorithm = cv2.KAZE_create()
#keypoint_algorithm = cv2.SIFT_create()


# Find the keypoints and compute the descriptors for input and training-set image
keypoints1, descriptors1 = keypoint_algorithm.detectAndCompute(image1, None)
keypoints2, descriptors2 = keypoint_algorithm.detectAndCompute(image2, None)


descriptor_matching_start_time = time.time()


# FLANN parameters
FLANN_INDEX_KDTREE = 1

index_params = dict(algorithm = FLANN_INDEX_KDTREE,
                    trees = 5)

search_params = dict(checks = 50)

# Convert to float32
descriptors1 = np.float32(descriptors1)
descriptors2 = np.float32(descriptors2)


save_reload_time = time.time()

save_format = 1

if save_format == 1:
    with h5py.File('descriptor1.h5', 'w') as hf:
        hf.create_dataset('data',  data=descriptors1)
    with h5py.File('descriptor1.h5', 'r') as hf:
        descriptors1 = hf['data'][:]

    with h5py.File('descriptor2.h5', 'w') as hf:
        hf.create_dataset('data',  data=descriptors2)
    with h5py.File('descriptor2.h5', 'r') as hf:
        descriptors2 = hf['data'][:]
else:
    np.save('descriptor1.npy', descriptors1)
    np.save('descriptor2.npy', descriptors2)
    descriptors1 = np.load('descriptor1.npy')
    descriptors2 = np.load('descriptor2.npy')


print("SAVE RELOAD TIME", time.time() - save_reload_time)


# Create FLANN object
FLANN = cv2.FlannBasedMatcher(indexParams = index_params,
                              searchParams = search_params)

# Matching descriptor vectors using FLANN Matcher
matches = FLANN.knnMatch(queryDescriptors = descriptors1,
                         trainDescriptors = descriptors2,
                         k = 2)

# Lowe's ratio test
ratio_thresh = 0.7

# "Good" matches
good_matches = []

# Filter matches
for m, n in matches:
    if m.distance < ratio_thresh * n.distance:
        good_matches.append(m)


print("GOOD MATCHES", len(good_matches))

print("DESCRIPTOR MATCHING TIME: ", time.time() - descriptor_matching_start_time)
print("OVERALL TIME: ", time.time() - overall_start_time)


# Draw only "good" matches
output = cv2.drawMatches(img1 = image1,
                         keypoints1 = keypoints1,
                         img2 = image2,
                         keypoints2 = keypoints2,
                         matches1to2 = good_matches,
                         outImg = None,
                         flags = cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

plt.imshow(output)
plt.show()
