import numpy as np
import cv2
from Camera import Camera
from ImageProcesing import *
from PerspectiveTransform import PerspectiveTransform
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from Lane import  Lane


class LaneDetector:
    # Read in a thresholded image
    # warped = mpimg.imread('warped_example.jpg')
    # window settings
    window_width = 50
    window_height = 80  # Break image into 9 vertical layers since image height is 720
    margin = 100  # How much to slide left and right for searching
    n_frames = 10 #number of frames to be saved for smoothing

    def __init__(self,debug = False):
        # camera object to calibrate the camera
        self.camera = Camera('camera_cal')
        #perspective transform object to make transformations
        self.pT = PerspectiveTransform()
        # left line
        self.left_line  = None
        #right line
        self.right_line = None
        # flag to enable/disable debugging
        self.debug = debug

    def draw_lanes_info(self,image):

        font = cv2.FONT_HERSHEY_SIMPLEX
        left_roc = "Roc: {0:.2f}m".format(self.left_line.calculate_radius())
        cv2.putText(image, left_roc, (10, 650), font, 1, (255, 255, 255), 2)
        right_roc = "Roc: {0:.2f}m".format(self.right_line.calculate_radius())
        cv2.putText(image, right_roc, (1020, 650), font, 1, (255, 255, 255), 2)

        # Write the x coords for each lane
        left_coord = "X  : {0:.2f}".format(self.left_line.line_base_pos)
        cv2.putText(image, left_coord, (10, 700), font, 1, (255, 255, 255), 2)
        right_coord = "X  : {0:.2f}".format(self.right_line.line_base_pos)
        cv2.putText(image, right_coord, (1020, 700), font, 1, (255, 255, 255), 2)

        # Write dist from center
        perfect_center = 1280 / 2.
        lane_x = self.right_line.line_base_pos - self.left_line.line_base_pos
        center_x = (lane_x / 2.0) + self.left_line.line_base_pos
        cms_per_pixel = 370.0 / lane_x  # US regulation lane width = 3.7m
        dist_from_center = (center_x - perfect_center) * cms_per_pixel
        dist_text = "Dist from Center: {0:.2f} cms".format(dist_from_center)
        cv2.putText(image, dist_text, (450, 50), font, 1, (255, 255, 255), 2)
        return image

    def draw_lanes(self,undistorted_image,warped):

        poly_left = np.poly1d(self.left_line.current_fit)
        poly_right = np.poly1d(self.right_line.current_fit)
        left_fitx = np.zeros(20)
        right_fitx = np.zeros(20)
        ploty  = np.zeros(20)
        for i in range(20):
            pixels_per_step = undistorted_image.shape[0] // 20
            y_point = undistorted_image.shape[0] - i * pixels_per_step

            left_fitx[i] = poly_left(y_point)
            right_fitx[i] = poly_right(y_point)
            ploty[i] = y_point
        #self.calculate_radius()
        # Create an image to draw the lines on
        warp_zero = np.zeros_like(warped).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))
        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

        # Draw lane lines only if lane was detected this frame
        cv2.polylines(color_warp, np.int_([pts_left]), False, (0,0,255), thickness=20)
        cv2.polylines(color_warp, np.int_([pts_right]), False, (255,0,0), thickness=20)


        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        newwarp = self.pT.get_reverse_transform(color_warp)

        # Combine the result with the original image
        result = cv2.addWeighted(undistorted_image, 1, newwarp, 0.3, 0)

        return result


    def detect_lanes(self, image):
        binary_image = combined_threshold(image)
        binary_warped = self.pT.get_perpective_transform(binary_image)


        left_detected = False
        right_detected = False
        left_x = left_y = right_x = right_y = []
        if self.left_line is not None and self.right_line is not None:
            left_x , left_y = self.search_along_previous_lane(binary_warped,self.left_line.current_fit)
            right_x, right_y = self.search_along_previous_lane(binary_warped, self.right_line.current_fit)
            left_detected, right_detected = self.validate_lane(left_x , left_y,right_x, right_y)

        if not left_detected:
            left_x , left_y = self.histogram_search(binary_warped,(250,binary_warped.shape[1]//2))

        if not right_detected:
            right_x, right_y = self.histogram_search(binary_warped,(binary_warped.shape[1]//2,binary_warped.shape[1]-250))

        if not left_detected or right_detected:
            left_detected, right_detected = self.validate_lane(left_x, left_y, right_x, right_y)

        if left_detected:
            if self.left_line is not None:
                self.left_line.update_params(left_x, left_y)
            else:
                self.left_line = Lane(n=self.n_frames, x=left_x, y=left_y)


        if right_detected:
            if self.right_line is not None:
                self.right_line.update_params(right_x,right_y)
            else:
                self.right_line = Lane(n=self.n_frames,x=right_x,y=right_y)

        output = image
        if self.left_line is not None and self.right_line is not None:
            output = self.draw_lanes(image,binary_warped)
            self.draw_lanes_info(output)
        return output

    def validate_lane(self,left_x , left_y,right_x, right_y):
        left_detected = False
        right_detected = False
        new_left = None
        new_right = None
        is_parallel = False
        distance_check = False

        # don't trust the data has less deetcted pixels
        if len(left_x) > 3:
            new_left = Lane(x=left_x, y=left_y)

        # don't trust the data has less deetcted pixels
        if len(right_x) > 3:
            new_right = Lane(x=right_x, y=right_y)


        # Chehck new found lanes with each other
        if new_left is not None and new_right is not None:
            is_parallel = new_left.is_parallel(new_right)
            distance_check = 360 < new_left.calculate_distance(new_right) < 550
            if is_parallel and distance_check:
                return True,True

        # check new lane againts old one
        if  self.left_line is not None and new_left is not None:
            is_parallel = new_left.is_parallel(self.left_line)
            distance_check = 360 < new_left.calculate_distance(self.left_line) < 550
            if is_parallel and distance_check:
                left_detected = True

        # check new line againts old values
        if self.right_line is not None and new_right is not None:
            is_parallel = new_right.is_parallel(self.right_line)
            distance_check = 360 < new_right.calculate_distance(self.right_line) < 550
            if is_parallel and distance_check:
                right_detected = True

        return left_detected, right_detected

    def check_lines(self,left_x , left_y,right_x, right_y):

        #don't trust the data has less deetcted pixels
        if len(left_x) < 3 or len(right_x) < 3:
            return False

        new_left = Lane(x=left_x,y=left_y)
        new_right = Lane(x=right_x, y=right_y)
        is_parallel = new_left.is_parallel(new_right)
        distance = new_left.calculate_distance(new_right)
        distance_check = 360 < distance < 550
        return (is_parallel and distance_check)


    def histogram_search(self,binary_warped,boundries):
        half = binary_warped[:,boundries[0]:boundries[1]]
        # Assuming you have created a warped binary image called "binary_warped"
        # Take a histogram of the bottom half of the image
        histogram = np.sum(half[half.shape[0] / 2:, :], axis=0)

        # Find the peak of the histogram
        # These will be the starting point for the line
        base = np.argmax(histogram[:])+boundries[0]
        # Choose the number of sliding windows
        nwindows = 9
        # Set height of windows
        window_height = np.int(binary_warped.shape[0] / nwindows)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Current positions to be updated for the winodw
        x_current = base
        # Set the width of the windows +/- margin
        margin = 100
        # Set minimum number of pixels found to recenter window
        minpix = 50
        # Create empty lists to receive left and right lane pixel indices
        lane_inds = []

        # Step through the windows one by one
        if self.debug :
            debug_image = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = binary_warped.shape[0] - (window + 1) * window_height
            win_y_high = binary_warped.shape[0] - window * window_height
            win_x_low = x_current - margin
            win_x_high = x_current + margin
            # Draw the windows on the visualization image
            if self.debug:
                # Create an output image to draw on and  visualize the result
                cv2.rectangle(debug_image, (win_x_low, win_y_low), (win_x_high, win_y_high), (0, 255, 0), 2)
            # Identify the nonzero pixels in x and y within the window
            good_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_x_low) & (
            nonzerox < win_x_high)).nonzero()[0]

            # Append these indices to the lists
            lane_inds.append(good_inds)

            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_inds) > minpix:
                x_current = np.int(np.mean(nonzerox[good_inds]))


        # Concatenate the arrays of indices
        lane_inds = np.concatenate(lane_inds)

        # Extract left and right line pixel positions
        x = nonzerox[lane_inds]
        y = nonzeroy[lane_inds]

        if self.debug:
            debug_image[nonzeroy[lane_inds], nonzerox[lane_inds]] = [255, 0, 0]
            plt.plot(x, y, color='yellow')
            plt.xlim(0, 1280)
            plt.ylim(720, 0)
            plt.show()
            plt.imshow(debug_image)
            plt.show()

        return x,y


    def search_along_previous_lane(self,binary_warped,line_fit):
        # Assume you now have a new warped binary image
        # from the next frame of video (also called "binary_warped")
        # It's now much easier to find line pixels!
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        margin = 100
        lane_inds = (
        (nonzerox > (line_fit[0] * (nonzeroy ** 2) + line_fit[1] * nonzeroy + line_fit[2] - margin)) & (
        nonzerox < (line_fit[0] * (nonzeroy ** 2) + line_fit[1] * nonzeroy + line_fit[2] + margin)))

        # Again, extract line pixel positions
        x = nonzerox[lane_inds]
        y = nonzeroy[lane_inds]

        if self.debug:
            # Fit a second order polynomial
            poly_fit = np.polyfit(x, y, 2)
            # Generate x and y values
            ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
            fitx = poly_fit[0] * ploty ** 2 + poly_fit[1] * ploty + poly_fit[2]

        return x,y

    def draw_hist_search_result(self,binary_warped,left_fit,right_fit,out_img,left_lane_inds,right_lane_inds,nonzerox,nonzeroy):
        # Generate x and y values for plotting
        ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
        left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
        plt.imshow(out_img)
        plt.plot(left_fitx, ploty, color='yellow')
        plt.plot(right_fitx, ploty, color='yellow')
        plt.xlim(0, 1280)
        plt.ylim(720, 0)
        plt.show()

    def process_image(self,image):
        undistorted_image = self.camera.undistort(image)
        output = self.detect_lanes(undistorted_image)

        return output

    ###########################################################################################################
    # sliding window search methods (currently not used on the pipeleine)
    # can be used in parallel with the histogram search to have better detection

    # method for locating the lanes
    # param image: undistordet transformed binary masked image
    def sliding_window_search(self,image):
        binary_image = combined_threshold(image)
        transformed_image = self.pT.get_perpective_transform(binary_image)
        window_centroids = self.find_window_centroids(transformed_image, self.window_width, self.window_height, self.margin)
        if len(window_centroids) > 0:
            # Points used to draw all the left and right windows
            l_points = np.zeros_like(transformed_image)
            r_points = np.zeros_like(transformed_image)
            l_xpoints = np.arange(len(window_centroids))
            ypoints = np.arange(len(window_centroids))
            r_xpoints = np.arange(len(window_centroids))
            # Go through each level and draw the windows
            for level in range(0, len(window_centroids)):
                # Window_mask is a function to draw window areas
                l_xpoints[level] = window_centroids[level][0]
                r_xpoints[level] = window_centroids[level][1]
                ypoints[level] = 720 - self.window_height * level - self.window_height / 2
                l_mask = self.window_mask(transformed_image, window_centroids[level][0], level)
                r_mask = self.window_mask(transformed_image, window_centroids[level][1], level)
                # Add graphic points from window mask here to total pixels found
                l_points[(l_points == 255) | ((l_mask == 1))] = 255
                r_points[(r_points == 255) | ((r_mask == 1))] = 255

            # Fit a second order polynomial to pixel positions in each fake lane line
            left_fit = np.polyfit(ypoints, l_xpoints, 2)
            left_fitx = left_fit[0] * ypoints ** 2 + left_fit[1] * ypoints + left_fit[2]

            right_fit = np.polyfit(ypoints, r_xpoints, 2)
            right_fitx = right_fit[0] * ypoints ** 2 + right_fit[1] * ypoints + right_fit[2]
            #draw_image = self.draw(image, transformed_image, left_fitx, right_fitx, ypoints)
            #output = self.calculate_radius(draw_image, left_fitx, right_fitx, ypoints, l_xpoints[0], r_xpoints[0])
        # If no window centers found, just display orginal road image
        else:
            output = np.array(cv2.merge((image, image, image)), np.uint8)

        return output

    def window_mask(self, img_ref, center, level):
        output = np.zeros_like(img_ref)
        output[int(img_ref.shape[0] - (level + 1) * self.window_height):int(img_ref.shape[0] - level * self.window_height),
        max(0, int(center - self.window_width / 2)):min(int(center +  self.window_width  / 2), img_ref.shape[1])] = 1
        return output

    def find_window_centroids(self,image, window_width, window_height, margin):
        window_centroids = []  # Store the (left,right) window centroid positions per level
        window = np.ones(window_width)  # Create our window template that we will use for convolutions

        # First find the two starting positions for the left and right lane by using np.sum to get the vertical image slice
        # and then np.convolve the vertical image slice with the window template

        # Sum quarter bottom of image to get slice, could use a different ratio
        l_sum = np.sum(image[int(3 * image.shape[0] / 4):, :int(image.shape[1] / 2)], axis=0)
        l_center = np.argmax(np.convolve(window, l_sum)) - window_width / 2
        r_sum = np.sum(image[int(3 * image.shape[0] / 4):, int(image.shape[1] / 2):], axis=0)
        r_center = np.argmax(np.convolve(window, r_sum)) - window_width / 2 + int(image.shape[1] / 2)

        # Add what we found for the first layer
        window_centroids.append((l_center, r_center))

        for level in range(1, (int)(image.shape[0] / window_height)):
            # convolve the window into the vertical slice of the image
            image_layer = np.sum(
                image[int(image.shape[0] - (level + 1) * window_height):int(image.shape[0] - level * window_height), :],
                axis=0)
            conv_signal = np.convolve(window, image_layer)
            # Find the best left centroid by using past left center as a reference
            # Use window_width/2 as offset because convolution signal reference is at right side of window, not center of window
            offset = window_width / 2
            l_min_index = int(max(l_center + offset - margin, 0))
            l_max_index = int(min(l_center + offset + margin, image.shape[1]))
            l_center = np.argmax(conv_signal[l_min_index:l_max_index]) + l_min_index - offset
            # if new center is far away from the previous one discard the new center and use the old one
            if abs(l_center - window_centroids[level - 1][0]) > window_width:
                l_center = window_centroids[level - 1][0]
            # Find the best right centroid by using past right center as a reference
            r_min_index = int(max(r_center + offset - margin, 0))
            r_max_index = int(min(r_center + offset + margin, image.shape[1]))
            r_center = np.argmax(conv_signal[r_min_index:r_max_index]) + r_min_index - offset
            # if new center is far away from the previous one discard the new center and use the old one
            if abs(r_center - window_centroids[level - 1][1]) > window_width:
                r_center = window_centroids[level - 1][1]
            # Add what we found for that layer
            window_centroids.append((l_center, r_center))

        return window_centroids


laneDetector = LaneDetector()
import os
from moviepy.editor import VideoFileClip

white_output = 'project_video_out.mp4'  # New video
os.remove(white_output)
clip1 = VideoFileClip('project_video.mp4')  # .subclip(21.00,25.00) # project video
# clip = VideoFileClip("myHolidays.mp4", audio=True).subclip(50,60)
white_clip = clip1.fl_image(laneDetector.process_image)  # NOTE: this function expects color images!!
white_clip.write_videofile(white_output, audio=False)

# import glob
# import time
# images = glob.glob('test_images/*.jpg')
#
# for image_path in images:
#     image = mpimg.imread(image_path)
#
#     t = time.time()
#     output = laneDetector.process_image(image)
#     t2 = time.time()
#     print(round(t2 - t, 2), 'procesing time ')
#
#     # t = time.time()
#     # YCrCb_result = train_test_model(ycrcb_parameters, do_training=False, model_path=YCrCb_model_path, image=image)
#     # t2 = time.time()
#     # print(round(t2 - t, 2), 'procesing time for YCrCb.')
#
#     fig = plt.figure()
#     plt.subplot(121)
#     plt.imshow(image)
#     plt.title('Original Image .')
#     plt.subplot(122)
#     plt.imshow(output)
#     plt.title('Lane.')
#     fig.tight_layout()
#     plt.show()