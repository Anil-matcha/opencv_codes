import cv2, os
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
import numpy as np

extractor = cv2.AKAZE_create()

def build_histogram(descriptor_list, cluster_alg):
    histogram = np.zeros(len(cluster_alg.cluster_centers_))
    cluster_result =  cluster_alg.predict(descriptor_list)
    for i in cluster_result:
        histogram[i] += 1.0
    return histogram

def gray(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray	
	
def features(image, extractor):
    keypoints, descriptors = extractor.detectAndCompute(image, None)
    return keypoints, descriptors

kmeans = KMeans(n_clusters = 800)	
preprocessed_image = []
files = [x for x in os.listdir() if "jpg" in x]
print(files)
images = [cv2.imread(img) for img in files]
descriptor_list = np.array([])
for image in images:
	image = gray(image)
	keypoint, descriptor = features(image, extractor)
	if len(descriptor_list) == 0:
		descriptor_list = np.array(descriptor)
	else:
		descriptor_list = np.vstack((descriptor_list, descriptor))
kmeans.fit(descriptor_list)	  
for image in images:
      image = gray(image)
      keypoint, descriptor = features(image, extractor)
      if (descriptor is not None):
          histogram = build_histogram(descriptor, kmeans)
          preprocessed_image.append(histogram)	

data = cv2.imread("book1.jpg")
data = gray(data)
keypoint, descriptor = features(data, extractor)
histogram = build_histogram(descriptor, kmeans)
neighbor = NearestNeighbors(n_neighbors = 5)
neighbor.fit(preprocessed_image)
dist, result = neighbor.kneighbors([histogram])
print([files[i] for i in result[0]])