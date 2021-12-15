# Depth-Estimation-using-Stereo-Vision

Uses the following data:
* raw non rectified iamges for city images [Found here](https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_26_drive_0001/2011_09_26_drive_0001_extract.zip)
* raw non rectified images for residential images [Found here](https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_26_drive_0001/2011_09_26_drive_0046_extract.zip
* ground truth depth data of lidar projected into carema frame [Found Here](https://s3.eu-central-1.amazonaws.com/avg-kitti/data_depth_velodyne.zip)
* camera calibriation files [Found Here](https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_26_calib.zip)

The python script will take each raw image pair and rectify it, then calculate a disparity map using two methods:
* SGMB - https://docs.opencv.org/3.4/d1/d9f/classcv_1_1stereo_1_1StereoBinarySGBM.html
* MeanShift Segmentation - Followed by cluster matching 

The depth value calculated for each image pair is the compared against the lidar data projected into the camera frame to get an raw error value. 
