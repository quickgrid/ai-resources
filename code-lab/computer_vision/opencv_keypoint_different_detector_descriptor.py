"""Opencv keypoint matching with different algorithm for detector and descriptors.

Not all combinations of detectors and descriptors work well together.
"""

import cv2
import matplotlib.pyplot as plt
import time

img1 = cv2.imread(r'imgs/1.png', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread(r'imgs/8.png', cv2.IMREAD_GRAYSCALE)

start1 = time.time()

kp_detector = cv2.BRISK_create()
#kp_detector = cv2.AgastFeatureDetector_create()
#kp_detector = cv2.KAZE_create()
#kp_detector = cv2.AKAZE_create()
#kp_detector = cv2.ORB_create()
#kp_detector = cv2.FastFeatureDetector_create()
#kp_detector = cv2.GFTTDetector_create()

descriptor_extractor = cv2.SIFT_create()
#descriptor_extractor = cv2.KAZE_create()
#descriptor_extractor = cv2.AKAZE_create()


keypoints1 = kp_detector.detect(img1)
keypoints2 = kp_detector.detect(img2)

# print(keypoints1)
# print(keypoints2)

keypoints1, descriptors1 = descriptor_extractor.compute(img1, keypoints1)
keypoints2, descriptors2 = descriptor_extractor.compute(img2, keypoints2)

# print(img1_kp)
# print(img1_desc)

# FLANN parameters
FLANN_INDEX_KDTREE = 1

index_params = dict(algorithm = FLANN_INDEX_KDTREE,
                    trees = 5)

search_params = dict(checks = 50)

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

end1 = time.time()
print("FPS:", (1. / float(end1 - start1)))

#img3 = cv2.drawMatchesKnn(img1,img1_kp,img2,img2_kp,matches,None,**draw_params)

output = cv2.drawMatches(img1 = img1,
                         keypoints1 = keypoints1,
                         img2 = img2,
                         keypoints2 = keypoints2,
                         matches1to2 = good_matches,
                         outImg = None,
                         flags = cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
plt.imshow(output,),plt.show()
