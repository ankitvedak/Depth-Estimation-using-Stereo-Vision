from sklearn.cluster import MeanShift, estimate_bandwidth
import numpy as np
import cv2

original_image = cv2.imread("left_img.png")

original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

blurimg = cv2.medianBlur(original_image, 5)

flat_image = blurimg.reshape((-1,3))
flat_image = np.float32(flat_image) 
bandwidth = estimate_bandwidth(flat_image, quantile=.06, n_samples=3000)
ms = MeanShift(bandwidth=bandwidth, max_iter=800, bin_seeding=True)
ms.fit(flat_image)
labeled = ms.labels_

segments = np.unique(labeled)
print('Number of segments: ', segments.shape[0])
print("cluster centroids", ms.cluster_centers_)
print("length of centroid matrix", len(ms.cluster_centers_))

# get the average color of each segment
total = np.zeros((segments.shape[0], 3), dtype=float)
count = np.zeros(total.shape, dtype=float)
for i, label in enumerate(labeled):
    total[label] = total[label] + flat_image[i]
    count[label] += 1
avg = total/count
avg = np.uint8(avg)

# cast the labeled image into the corresponding average color
res = avg[labeled]
result = res.reshape((original_image.shape))

cv2.imshow('result',result)
cv2.imwrite("left_image.png",result)
cv2.waitKey(0)
cv2.destroyAllWindows()


original_image = cv2.imread("right_img.png")

original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
blurimg = cv2.medianBlur(original_image, 5)
flat_image = blurimg.reshape((-1,3))
flat_image = np.float32(flat_image) 
bandwidth = estimate_bandwidth(flat_image, quantile=.06, n_samples=3000)
ms = MeanShift(bandwidth=bandwidth, max_iter=800, bin_seeding=True)
ms.fit(flat_image)
labeled = ms.labels_

segments = np.unique(labeled)
print('Number of segments: ', segments.shape[0])
print("cluster centroids", ms.cluster_centers_)
print("length of centroid matrix", len(ms.cluster_centers_))

# get the average color of each segment
total = np.zeros((segments.shape[0], 3), dtype=float)
count = np.zeros(total.shape, dtype=float)
for i, label in enumerate(labeled):
    total[label] = total[label] + flat_image[i]
    count[label] += 1
avg = total/count
avg = np.uint8(avg)

# cast the labeled image into the corresponding average color
res = avg[labeled]
result = res.reshape((original_image.shape))

cv2.imshow('result',result)
cv2.imwrite("right_image.png",result)
cv2.waitKey(0)
cv2.destroyAllWindows()


# print("cluster labels", clustering.labels_)

# cv2.imwrite('scikit_clustering.png', clustering.labels_)
