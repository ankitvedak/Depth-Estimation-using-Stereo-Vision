import cv2
import sys
import pymeanshift as pms
original_image = cv2.imread("example-orig.jpg")

(segmented_image, labels_image, number_regions) = pms.segment(original_image, spatial_radius=6, 
                                                              range_radius=4.5, min_density=50)

cv2.imwrite('segmented_image.png', segmented_image)
cv2.imwrite('labels_image.png', labels_image)
print("number of regions: " , number_regions)

