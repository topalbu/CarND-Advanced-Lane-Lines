import cv2
import glob
import pickle
import os
import numpy as np
#Class for camera calibratiom
class Camera:
    def __init__(self, cal_images_path = 'camera_cal'):
        self.images = glob.glob(cal_images_path +'/calibration*.jpg')
        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        self.objp = np.zeros((6 * 9, 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)
        # Arrays to store object points and image points from all the images.
        self.objpoints = []  # 3d points in real world space
        self.imgpoints = []  # 2d points in image plane.
        self.mtx = None
        self.dist = None
        self.image_shape = None

    def calibrate(self,image_shape):
        # Step through the list and search for chessboard corners
        self.image_shape = image_shape
        for fname in self.images:
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Find the chessboard corners
            ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)

            # If found, add object points, image points
            if ret == True:
                self.objpoints.append(self.objp)
                self.imgpoints.append(corners)

        #name the calibration file according to the image shape
        file_name = 'calibration_' + str(image_shape[1]) + str(image_shape[0]) + '.p'

        # if the calibration parameters were calculated and saved before read the calibration parameters from file
        if os.path.isfile(file_name):
            with open(file_name, mode='rb') as f:
                calibtate = pickle.load(f)
            self.mtx, self.dist = calibtate['mtx'], calibtate['dist']
        else: # if the calibration hasnt done yet for the given image and shape calibrate the camera
            # Do camera calibration given object points and image points
            ret, self.mtx, self.dist, rvecs, tvecs = cv2.calibrateCamera(self.objpoints, self.imgpoints, (image_shape[1],image_shape[0]), None, None)
            # Save the camera calibration result for later use (we won't worry about rvecs / tvecs)
            dist_pickle = {}
            dist_pickle["mtx"] = self.mtx
            dist_pickle["dist"] = self.dist
            pickle.dump(dist_pickle, open(file_name, "wb"))

    def undistort(self, img):
        # method to undistort the given image
        # check if the calibratuion parameters calulated if not calibrate the camera
        if self.mtx is None or self.dist is None:
            self.calibrate(img.shape)
        # check if the calibratuion parameters calulated for the image size if not re calculate again
        if img.shape != self.image_shape:
            self.calibrate(img.shape)
        #undistort image and return
        return cv2.undistort(img, self.mtx, self.dist, None, self.mtx)

