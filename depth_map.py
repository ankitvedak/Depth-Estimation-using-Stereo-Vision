import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import PIL.Image as pil
from sklearn.cluster import MeanShift, estimate_bandwidth

import matplotlib.patches as mpatches
from skimage.measure import label, regionprops
from skimage.morphology import closing, square
from skimage.color import label2rgb
from skimage.feature import match_template

def rectify_images(imgL, imgR, I, E):
	sz = (len(imgL[0]), len(imgL))	

	# re-rectify using scaled image size
	P1 = np.zeros((3,4))
	P2 = np.zeros((3,4))
	R1 = np.zeros((3,3))
	R2 = np.zeros((3,3))
	Q = np.zeros((4,4))
	cv2.stereoRectify(I['M1'], I['D1'], I['M2'], I['D2'], sz, E['R'], E['T'], R1, R2, P1, P2, Q, newImageSize=(1242, 375))

	#print("P1:", P1, "\nP2",P2, "\nR1", R1, "\nR2", R2, "\nQ", Q)

	# apply rectification
	map11, map12 = cv2.initUndistortRectifyMap(I['M1'], I['D1'], R1, P1,  (1242, 375), cv2.CV_32FC1)
	map21, map22 = cv2.initUndistortRectifyMap(I['M2'], I['D2'], R2, P2,  (1242, 375), cv2.CV_32FC1)

	imgL_r = cv2.remap(imgL, map11, map12, cv2.INTER_LINEAR);
	imgR_r = cv2.remap(imgR, map21, map22, cv2.INTER_LINEAR);

	return imgL_r, imgR_r, Q

def load_camera_calibrations(calibration_file_path):
	calibration = {}
	with open(calibration_file_path, 'r') as calibration_file:
		for line in calibration_file:
			split_line = line.strip().split(":")
			calibration[split_line[0]] = split_line[1].strip().split(" ")
	#print(calibration)
	
	m1 = np.reshape(list(map(lambda x: float(x), calibration['K_02'])), (3,3))
	m2 = np.reshape(list(map(lambda x: float(x), calibration['K_03'])), (3,3))

	d1 = np.reshape(list(map(lambda x: float(x), calibration['D_02'])), (1,5))
	d2 = np.reshape(list(map(lambda x: float(x), calibration['D_03'])), (1,5))

	world_t_c1r = np.reshape(list(map(lambda x: float(x), calibration['R_02'])), (3,3))
	world_t_c1t = list(map(lambda x: float(x), calibration['T_02']))

	world_t_c2r = np.reshape(list(map(lambda x: float(x), calibration['R_03'])), (3,3))
	world_t_c2t = list(map(lambda x: float(x), calibration['T_03']))
	#print(m1, "\n", m2,"\n", d1,"\n", d2, "\n",world_t_c1r,"\n", world_t_c1t,"\n", world_t_c2r,"\n", world_t_c2t)

	world_t_c1 = np.zeros((4,4))
	world_t_c1[:3, :3] = world_t_c1r
	world_t_c1[:3, 3] = world_t_c1t
	world_t_c1[3, :] = [ 0, 0, 0, 1]

	world_t_c2 = np.zeros((4,4))
	world_t_c2[:3, :3] = world_t_c2r
	world_t_c2[:3, 3] = world_t_c2t
	world_t_c2[3, :] = [ 0, 0, 0, 1]	
	#print("world_t_c1 \n", world_t_c1, "\nworld_t_c2\n", world_t_c2)
	
	# get transformations between cameras so p_c1 * inv(r1) = world * r2 = p_c2 
	# so r = r2 * inv(r1)
	c1_t_c2 = np.zeros((4,4))
	c1_t_c2 = world_t_c2 @ np.linalg.inv(world_t_c1) 
	#print("c1_t_c2\n", c1_t_c2)

	r =    c1_t_c2[0:3, 0:3]
	t = c1_t_c2[0:3, 3]
	#print("R\n", r)
	#print("T\n", t)

	I = { 'M1': m1, 'M2': m2, 'D1': d1, 'D2': d2}
	E = {'R':r, 'T':t } 

	return I,E

def getDisparity_SGBM(left_image, right_image):
	sad_window = 3
	num_disparities = 128
	block_size = 3
	matcher = cv2.StereoSGBM_create(numDisparities=num_disparities,
                                    minDisparity=0,
                                    blockSize=block_size,
                                    P1 = 4 * 1 * block_size ** 2,
                                    P2 = 32 * 1 * block_size ** 2,
                                    mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
	return matcher.compute(left_image, right_image)

def getDisparity_MeanShift(left_image, right_image):
	#convert to gray because scikit-image hates color 
	left_gray = cv2.cvtColor(left_image, cv2.COLOR_BGR2GRAY)
	right_gray = cv2.cvtColor(right_image, cv2.COLOR_BGR2GRAY)

	sgemented_left, left_clusters = segment_image(left_image)
	segmented_right, right_clusters = segment_image(right_image)

	plt.figure(1)
	plt.imshow(sgemented_left)
	plt.figure(2)
	plt.imshow(segmented_right)
	plt.show()

	label_to_dist = {}
	#print(left_clusters, right_clusters)
	disp = np.zeros((375,1242))
	left_clusters = sorted(left_clusters,  key=lambda x:x.area, reverse=True)
	for n in range(0,len(left_clusters)):
		dist = 0  
		label = left_clusters[n].label
		result = match_template(right_gray, left_clusters[n].intensity_image)
		ij = np.unravel_index(np.argmax(result), result.shape)
		x, y = ij[::-1]
		if x and y: 
			dist = x -left_clusters[n].centroid[0] 
		disp[left_clusters[n].slice] = dist 

	#print(len(label_to_dist))
	return disp.astype(np.float32)


def segment_image(image):
	blurimg = cv2.medianBlur(image, 5)
	flat_image = blurimg.reshape((-1,3))
	flat_image = np.float32(flat_image) 
	bandwidth = estimate_bandwidth(flat_image, quantile=.06, n_samples=3000)
	ms = MeanShift(bandwidth=bandwidth, max_iter=800, bin_seeding=True)
	ms.fit(flat_image)
	labeled = ms.labels_

	segments = np.unique(labeled)
	#print('Number of segments: ', segments.shape[0])
	#print("cluster centroids", ms.cluster_centers_)
	#print("length of centroid matrix", len(ms.cluster_centers_))

	# get the average color of each segment
	total = np.zeros((segments.shape[0], 3), dtype=float)
	count = np.zeros(total.shape, dtype=float)
	#print(np.shape(labeled), np.shape(flat_image), np.shape(total))
	for i, label in enumerate(labeled):
	    total[label] += flat_image[i]
	    count[label] += 1
	avg = total/count
	avg = np.uint8(avg)

	# cast the labeled image into the corresponding average color
	res = avg[labeled]
	result = res.reshape((image.shape))

	labeled_image, centroids = find_centriods(result)
	return labeled_image, centroids

def find_centriods(segmented_image):
	gray = cv2.cvtColor(segmented_image, cv2.COLOR_BGR2GRAY)
	gray = closing(gray)
	labeled_image = label(gray)
	image_label_overlay = label2rgb(labeled_image, image=gray, bg_label=0)
	regions = regionprops(labeled_image, gray)

	filtered_regions = list(filter(lambda region: region.area > 10, regions))

	return image_label_overlay, filtered_regions

def calculate_error(depth_map, image_path):
	error = 0
	for filename in depth_map.keys(): 
		file_path = image_path + filename

		depth_gt = pil.open(file_path)
		depth_gt = np.array(depth_gt).astype(np.float32) / 256

		plt.figure(1)
		plt.imshow(depth_gt)
		plt.show()


		ground_truth = depth_gt *  0.54 #Scale the results by the baseline 
		computed = np.where(depth_map[filename][:, :, 2]<10000, depth_map[filename][:, :, 2], 0)
		computed = computed 

		diff = np.zeros((len(ground_truth)-1, len(ground_truth[0])-1))
		for j in range(0, len(ground_truth)-1):
			for i in range(0, len(ground_truth[0])-1):
				if (ground_truth[j][i] > 0 and computed[j][i] > 0 ):
					spot_e = np.abs(computed[j][i] - ground_truth[j][i])
					diff[j][i] = int(spot_e)
					error += spot_e

		#print("error:", error, "mean", np.mean(diff[diff > 0]), " ", np.std(diff[diff>0]))

	return error, np.mean(diff[diff>0]), np.std(diff[diff>0]) 

data_dir = "/home/BOSDYN/bvalentino/Documents/2011_09_26/"
raw_images_path = "/2011_09_26_drive_0001_extract/"
depth_images_path = "/2011_09_26_drive_0001_sync/proj_depth/groundtruth/"
config_file = "calib_cam_to_cam.txt"
depth_map_smgb = {} 
depth_map_mean = {}

if data_dir + config_file:
	# I.M1, I.D1, I.M2, I.D2, E.R, E.T
	I,E = load_camera_calibrations(data_dir + config_file)

count = 0

print("Filename", "SGBM Error", "Mean Shift Error\n")

for file_path in glob.glob(data_dir + "2011_09_26_drive_0046_sync/proj_depth/groundtruth/image_02/*"):
	if count >= 1:
		break 

	filename = os.path.basename(file_path)
	print(filename)
	img_left_raw = cv2.imread(data_dir + "2011_09_26_drive_0046_extract/image_02/data/" + filename)
	img_right_raw = cv2.imread(data_dir + "2011_09_26_drive_0046_extract/image_03/data/" + filename)

	img_left_rectified, img_right_rectified, Q = rectify_images(img_left_raw, img_right_raw, I, E);

	disparity_SGBM = getDisparity_SGBM(img_left_rectified, img_right_rectified)
	disparity_MeanShift = getDisparity_MeanShift(img_left_rectified, img_right_rectified)

	depth_sgbm = cv2.reprojectImageTo3D(disparity_SGBM, Q, handleMissingValues=True)
	depth_mean = cv2.reprojectImageTo3D(disparity_MeanShift, Q, handleMissingValues=True)

	plt.figure(1)
	plt.imshow(disparity_SGBM)
	plt.figure(2)
	plt.imshow(disparity_MeanShift)
	plt.figure(3)
	plt.imshow(depth_sgbm)
	plt.figure(4)
	plt.imshow(depth_mean)
	plt.figure(5)
	plt.imshow(img_left_rectified)
	plt.figure(6)
	plt.imshow(img_right_rectified)
	plt.show()


	depth_map_smgb[filename]  = depth_sgbm
	depth_map_mean[filename] = depth_mean

	count += 1

	sgbm_error, mean, sd = calculate_error(depth_map_smgb,  data_dir + "2011_09_26_drive_0046_sync/proj_depth/groundtruth/image_02/")
	mean_error, mean, sd = calculate_error(depth_map_mean, data_dir + "2011_09_26_drive_0046_sync/proj_depth/groundtruth/image_02/")

	print(filename, ",",sgbm_error,",", mean_error,"\n")

count = 0
for file_path in glob.glob(data_dir + depth_images_path + "image_02/*"):
	if count >= 1:
		break 

	filename = os.path.basename(file_path)
	print(filename)
	img_left_raw = cv2.imread(data_dir + "2011_09_26_drive_0001_extract/image_02/data/" + filename)
	img_right_raw = cv2.imread(data_dir + "2011_09_26_drive_0001_extract/image_03/data/" + filename)

	img_left_rectified, img_right_rectified, Q = rectify_images(img_left_raw, img_right_raw, I, E);

	disparity_SGBM = getDisparity_SGBM(img_left_rectified, img_right_rectified)
	disparity_MeanShift = getDisparity_MeanShift(img_left_rectified, img_right_rectified)

	depth_sgbm = cv2.reprojectImageTo3D(disparity_SGBM, Q, handleMissingValues=True)
	depth_mean = cv2.reprojectImageTo3D(disparity_MeanShift, Q, handleMissingValues=True)

	plt.figure(1)
	plt.imshow(disparity_SGBM)
	plt.figure(2)
	plt.imshow(disparity_MeanShift)
	plt.figure(3)
	plt.imshow(depth_sgbm)
	plt.figure(4)
	plt.imshow(depth_mean)
	plt.figure(5)
	plt.imshow(img_left_rectified)
	plt.figure(6)
	plt.imshow(img_right_rectified)
	plt.show()


	depth_map_smgb[filename]  = depth_sgbm
	depth_map_mean[filename] = depth_mean

	count += 1

	sgbm_error, mean, sd = calculate_error(depth_map_smgb, data_dir + depth_images_path + "image_02/")
	mean_error, mean, sd = calculate_error(depth_map_mean, data_dir + depth_images_path + "image_02/")

	print(filename, ",",sgbm_error,",", mean_error,"\n")


