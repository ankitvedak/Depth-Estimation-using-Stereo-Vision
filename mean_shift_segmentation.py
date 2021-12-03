import cv2
import sys
import pymeanshift as pms
original_image = cv2.imread("example-orig.jpg")

bgrimg = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY) 

(segmented_image, labels_image, number_regions) = pms.segment(bgrimg, spatial_radius=10, 
                                                              range_radius=10, min_density=300)

# img = cv2.cvtColor(segmented_image, cv2.COLOR_BGR2GRAY)                                                         

ret, thresh = cv2.threshold(segmented_image,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

connectivity = 8 

output = cv2.connectedComponentsWithStats(thresh, connectivity, cv2.CV_32S)

num_labels = output[0]

labels = output[1]

stats = output[2]

centroids = output[3]

cv2.imwrite('segmented_image.png', segmented_image)
cv2.imwrite('labels_image.png', labels_image)
cv2.imwrite('bgrimage.png', bgrimg)

print("number of regions: " , number_regions)
print("image with labels", labels)
print("centroids position", centroids)
print("centroid matrix size", len(centroids))

