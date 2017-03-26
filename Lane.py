import numpy as np
class Lane:
    def __init__(self,n=1,x=None,y=None):
        '''
        :param n: number of frames to be kept on memory
        :param x: x pixel values
        :param y: y pixel values
        '''
        # was the line detected in the last iteration?
        self.detected = False
        # number of frames to be saved
        self.n_frames = n
        # x values of the last n fits of the line
        self.recent_xfitted = []
        #number of pixels added for each frame
        self.n_pixel_per_frame = []
        #average x values of the fitted line over the last n iterations
        self.bestx = None
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        #polynomial of the best fit
        self.best_fit_poly = None
        #polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]
        #polynomial of the current fit
        self.current_fit_poly = None
        #radius of curvature of the line in some units
        self.radius_of_curvature = None
        #distance in meters of vehicle center from the line
        self.line_base_pos = None
        #x values for detected line pixels
        self.allx = None
        #y values for detected line pixels
        self.ally = None

        if x is not None and y is not None:
            self.update_params(x,y)

    def update_params(self, x, y):
        """
        Updates the line parameters
        :param x: list of x values
        :param y: list of y values
        """
        assert len(x) == len(y), 'x and y have to be the same size'

        self.allx = x
        self.ally = y
        self.line_base_pos = x[-1]
        self.n_pixel_per_frame.append(len(self.allx))
        self.recent_xfitted.extend(self.allx)

        if len(self.n_pixel_per_frame) > self.n_frames:
            n_x_to_remove = self.n_pixel_per_frame.pop(0)
            self.recent_xfitted = self.recent_xfitted[n_x_to_remove:]

        self.bestx = np.mean(self.recent_xfitted)

        self.current_fit = np.polyfit(self.ally, self.allx, 2)
        if self.best_fit is None:
            self.best_fit = self.current_fit
        else:
            self.best_fit = (self.best_fit * (self.n_frames - 1) + self.current_fit) / self.n_frames

        self.best_fit_poly = np.poly1d(self.best_fit)
        self.current_fit_poly = np.poly1d(self.current_fit)

    def is_parallel(self, other):
        '''
        Check if the two lines are parallel by comparing the coefficients
        :param other: other line to compare
        :return: True if the coefficients are similar
        '''
        is_parallel = np.abs(self.best_fit[0] - other.best_fit[0]) < 0.00003 and  np.abs(self.best_fit[1] - other.best_fit[1]) < 0.5
        return is_parallel

    def calculate_distance(self, other):
        """
        Calculate distance between two lines
        :param other: Line to calculate distance from
        :return: the distance from other line to self
        """
        return np.abs(self.best_fit_poly(719) - other.best_fit_poly(719))

    def calculate_radius(self):
        '''
        Calculates the radius of the polynom
        :return: calculated radius_of_curvature
        '''
        y_points  = np.array(np.linspace(0, 719, num=10))
        y_eval = np.max(y_points)
        x_points = np.zeros(len(y_points))

        #Calculate the points according to best fit polynomial
        for i in range(len(y_points)):
            x_points[i] = self.best_fit_poly(y_points[i])

        # Define conversions in x and y from pixels space to meters
        ym_per_pix = 30 / 720  # meters per pixel in y dimension
        xm_per_pix = 3.7 / 700  # meters per pixel in x dimension

        # Fit new polynomials to x,y in world space
        fit_cr = np.polyfit(y_points * ym_per_pix, x_points * xm_per_pix, 2)

        radius_of_curvature= ((1 + (2 * fit_cr[0] * y_eval * ym_per_pix + fit_cr[1]) ** 2) ** 1.5) / np.absolute(
            2 * fit_cr[0])

        return radius_of_curvature

    def check_curvature(self):
        '''
        Checks the lane curvature against the standards
        :param radius: curvature of the radious of the found lane
        :return: True if the curvature is in standards
        '''
        min_curvature = 900 # in meters
        max_curvature = 4500 # in meters
        return (min_curvature < self.calculate_radius() < max_curvature)
