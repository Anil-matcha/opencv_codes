# Standard imports
import cv2
import numpy as np;

params = cv2.SimpleBlobDetector_Params()
params.filterByConvexity = True
params.minConvexity = 0.1
params.filterByCircularity = True
params.minCircularity = 0.1
params.filterByArea = True
params.minArea = 100
params.filterByInertia = True
params.minInertiaRatio = 0.01
params.minThreshold = 10;
params.maxThreshold = 225;

im = cv2.imread("blob.jpg", cv2.IMREAD_GRAYSCALE) 
detector = cv2.SimpleBlobDetector_create(params)
keypoints = detector.detect(im)
im_with_keypoints = cv2.drawKeypoints(im, keypoints, np.array([]), (0,255,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imshow("Keypoints", im_with_keypoints)
cv2.waitKey(0)