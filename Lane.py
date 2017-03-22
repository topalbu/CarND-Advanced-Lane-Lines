import numpy as np
import cv2

class Line:
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False
        # x values of the last n fits of the line
        self.recent_xfitted = []
        #average x values of the fitted line over the last n iterations
        self.bestx = None
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        #polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]
        #radius of curvature of the line in some units
        self.radius_of_curvature = None
        #distance in meters of vehicle center from the line
        self.line_base_pos = None
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float')
        #x values for detected line pixels
        self.allx = None
        #y values for detected line pixels
        self.ally = None

class Lane:
    # Read in a thresholded image
    # warped = mpimg.imread('warped_example.jpg')
    # window settings
    window_width = 50
    window_height = 80  # Break image into 9 vertical layers since image height is 720
    margin = 100  # How much to slide left and right for searching


    def __init__(self,perspectiveTransformer):
        self.pT = perspectiveTransformer
        left_line  = Line()
        right_line = Line()

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
        # image[:,0:l_center-window_width] = 0
        # image[:,r_center+window_width:image.shape[1]] = 0
        # plt.imshow(image,cmap='gray')
        # plt.show()

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

    def calculate_radius(self,image, left_fitx, right_fitx, ploty, left_lane_coordinate, right_lane_coordinate):
        # Define y-value where we want radius of curvature
        # I'll choose the maximum y-value, corresponding to the bottom of the image
        y_eval = np.max(ploty)
        # left_curverad = ((1 + (2*left_fit[0]*y_eval + left_fit[1])**2)**1.5) / np.absolute(2*left_fit[0])
        # right_curverad = ((1 + (2*right_fit[0]*y_eval + right_fit[1])**2)**1.5) / np.absolute(2*right_fit[0])
        # print(left_curverad, right_curverad)
        # Define conversions in x and y from pixels space to meters
        ym_per_pix = 30 / 720  # meters per pixel in y dimension
        xm_per_pix = 3.7 / 700  # meters per pixel in x dimension
        # Fit new polynomials to x,y in world space
        left_fit_cr = np.polyfit(ploty * ym_per_pix, left_fitx * xm_per_pix, 2)
        right_fit_cr = np.polyfit(ploty * ym_per_pix, right_fitx * xm_per_pix, 2)
        # Calculate the new radii of curvature
        left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
            2 * left_fit_cr[0])
        right_curverad = (
                         (1 + (2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
            2 * right_fit_cr[0])
        # Now our radius of curvature is in meters
        # Example values: 632.1 m    626.2 m
        # Write the radius of curvature for each lane
        font = cv2.FONT_HERSHEY_SIMPLEX
        left_roc = "Roc: {0:.2f}m".format(left_curverad)
        cv2.putText(image, left_roc, (10, 650), font, 1, (255, 255, 255), 2)
        right_roc = "Roc: {0:.2f}m".format(right_curverad)
        cv2.putText(image, right_roc, (1020, 650), font, 1, (255, 255, 255), 2)

        # Write the x coords for each lane
        left_coord = "X  : {0:.2f}".format(left_lane_coordinate)
        cv2.putText(image, left_coord, (10, 700), font, 1, (255, 255, 255), 2)
        right_coord = "X  : {0:.2f}".format(right_lane_coordinate)
        cv2.putText(image, right_coord, (1020, 700), font, 1, (255, 255, 255), 2)

        # Write dist from center
        perfect_center = 1280 / 2.
        lane_x = right_lane_coordinate - left_lane_coordinate
        center_x = (lane_x / 2.0) + left_lane_coordinate
        cms_per_pixel = 370.0 / lane_x  # US regulation lane width = 3.7m
        dist_from_center = (center_x - perfect_center) * cms_per_pixel
        dist_text = "Dist from Center: {0:.2f} cms".format(dist_from_center)
        cv2.putText(image, dist_text, (450, 50), font, 1, (255, 255, 255), 2)
        return image

    def draw(self,undistorted_image, warped, left_fitx, right_fitx, ploty):
        # Create an image to draw the lines on
        warp_zero = np.zeros_like(warped).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        newwarp = self.Pt.get_reverse_transform(color_warp)
        # Combine the result with the original image
        result = cv2.addWeighted(undistorted_image, 1, newwarp, 0.3, 0)
        return result

    # method for locating the lanes
    # param image: undistordet transformed binary masked image
    def find_lanes(self,image):
        window_centroids = self.find_window_centroids(image, self.window_width, self.window_height, self.margin)
        if len(window_centroids) > 0:
            # Points used to draw all the left and right windows
            l_points = np.zeros_like(image)
            r_points = np.zeros_like(image)
            l_xpoints = np.arange(len(window_centroids))
            ypoints = np.arange(len(window_centroids))
            r_xpoints = np.arange(len(window_centroids))
            # Go through each level and draw the windows
            for level in range(0, len(window_centroids)):
                # Window_mask is a function to draw window areas
                l_xpoints[level] = window_centroids[level][0]
                r_xpoints[level] = window_centroids[level][1]
                ypoints[level] = 720 - self.window_height * level - self.window_height / 2
                l_mask = self.window_mask(image, window_centroids[level][0], level)
                r_mask = self.window_mask(image, window_centroids[level][1], level)
                # Add graphic points from window mask here to total pixels found
                l_points[(l_points == 255) | ((l_mask == 1))] = 255
                r_points[(r_points == 255) | ((r_mask == 1))] = 255

            # Fit a second order polynomial to pixel positions in each fake lane line
            left_fit = np.polyfit(ypoints, l_xpoints, 2)
            left_fitx = left_fit[0] * ypoints ** 2 + left_fit[1] * ypoints + left_fit[2]

            right_fit = np.polyfit(ypoints, r_xpoints, 2)
            right_fitx = right_fit[0] * ypoints ** 2 + right_fit[1] * ypoints + right_fit[2]
            draw_image = self.draw(image, image, left_fitx, right_fitx, ypoints)
            output = self.calculate_radius(draw_image, left_fitx, right_fitx, ypoints, l_xpoints[0], r_xpoints[0])

        # If no window centers found, just display orginal road image
        else:
            output = np.array(cv2.merge((transformed_image, transformed_image, transformed_image)), np.uint8)

    def process_image2(image):
        camera = Camera()
        undistorted_image = camera.undistort(image)
        binary_image = combined_threshold(undistorted_image)
        transformed_image = pt.get_perpective_transform(binary_image)
        window_centroids = find_window_centroids(transformed_image, window_width, window_height, margin)
        # If we found any window centers
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
                ypoints[level] = 720 - window_height * level - window_height / 2
                l_mask = window_mask(window_width, window_height, transformed_image, window_centroids[level][0], level)
                r_mask = window_mask(window_width, window_height, transformed_image, window_centroids[level][1], level)
                # Add graphic points from window mask here to total pixels found
                l_points[(l_points == 255) | ((l_mask == 1))] = 255
                r_points[(r_points == 255) | ((r_mask == 1))] = 255

            # Fit a second order polynomial to pixel positions in each fake lane line
            left_fit = np.polyfit(ypoints, l_xpoints, 2)
            left_fitx = left_fit[0] * ypoints ** 2 + left_fit[1] * ypoints + left_fit[2]

            right_fit = np.polyfit(ypoints, r_xpoints, 2)
            right_fitx = right_fit[0] * ypoints ** 2 + right_fit[1] * ypoints + right_fit[2]
            draw_image = draw(undistorted_image, transformed_image, left_fitx, right_fitx, ypoints)
            output = calculate_radius(draw_image, left_fitx, right_fitx, ypoints, l_xpoints[0], r_xpoints[0])

        # If no window centers found, just display orginal road image
        else:
            output = np.array(cv2.merge((transformed_image, transformed_image, transformed_image)), np.uint8)

        return output

