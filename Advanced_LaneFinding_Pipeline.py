import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import pickle
import math
import re

# Custom Libs
import preprocessing_frames

# custom vars
calibration_dir = "camera_cal"
test_imgs_dir = "test_images"
output_imgs_dir = "output_images"
output_videos_dir = "output_videos"
test_imgs_undist_dir = "test_images_undistorted"

# Read in the saved camera matrix and dist coeffs
dist_pickle = pickle.load(open("camera_cal/wide_dist_pickle.p", "rb"))
mtx = dist_pickle["mtx"]
dist = dist_pickle["dist"]

# parameters
k_size = 15  # kernel_size
sobel_threshold_xy = (20, 120)  # Threshold value
mag_threshold = (80, 200)  # Threshold value
dir_threshold = (np.pi / 4, np.pi / 2)  # Threshold value
(height_y, width_x) = (719, 1279)  # Height and width of frames
src_pts = np.array([[210, height_y], [595, 450], [690, 450],
                    [1110, height_y]], np.float32)  # Source points for perspective transform
dst_pts = np.array([[200, height_y], [200, 0],
                    [1000, 0], [1000, height_y]], np.float32)  # Destination points for perspective transform

# HYPER-PARAMETERS
# Choose the number of sliding windows
nwindows = 9
# Set the width of the windows +/- margin
margin = 100
# Set minimum number of pixels found to recenter window
minpix = 100
lane_size_metres = (32, 3.7)
lane_width = 1000 - 200  # 1000 -> right lane position, 200 -> left lane pos. after psp transform
lane_center = (1000 + 200) / 2


# Class LaneLine() contains the attributes of the Lane Lines
class LaneLine:
    def __init__(self):
        self.polynomial_coeff = None
        self.line_fit_x = None
        self.non_zero_x = []
        self.non_zero_y = []
        self.windows = []


# Class AdvancedLaneLineDetector() processes the complete Lane Finding Function
class AdvancedLaneLineDetector:
    def __init__(self):
        self.previous_left_lane_line = None
        self.previous_right_lane_line = None
        self.img_dimensions = (720, 1280)
        self.ploty = np.linspace(0, self.img_dimensions[0] - 1, self.img_dimensions[0])
        self.real_world_lane_size = lane_size_metres
        self.ym_per_px = self.real_world_lane_size[0] / self.img_dimensions[0]
        self.xm_per_px = self.real_world_lane_size[1] / lane_width

    def process_frame(self, img):
        # img = mpimg.imread(fname)
        undist_img = self.undistort_img(img)
        preprocessed_frame = self.apply_transformation(undist_img)
        warped_frame = self.apply_perspective_transform(preprocessed_frame, src_pts, dst_pts)
        left_lane, right_lane, out_img = self.detect_lane_lines_pixels(warped_frame)
        left_curve, right_curve, center_offset = self.measure_curvature(left_lane, right_lane)
        lane_area_img = self.draw_lane_area(out_img, undist_img, left_lane, right_lane)
        processed_frame = self.draw_lane_curvature_text(lane_area_img, left_curve, right_curve, center_offset)
        print(left_curve, right_curve, center_offset)

        self.previous_left_lane_line = left_lane
        self.previous_right_lane_line = right_lane

        return processed_frame

    def undistort_img(self, img):
        return cv2.undistort(img, mtx, dist, None, mtx)

    def apply_transformation(self, image):
        img = np.copy(image)
        proc = preprocessing_frames.PreprocessingPipeline(k_size, sobel_threshold_xy, mag_threshold, dir_threshold)

        # Apply each of the thresholding functions
        gradx = proc.abs_sobel_thresh(img, orient='x')
        grady = proc.abs_sobel_thresh(img, orient='y')
        mag_binary = proc.mag_threshold(img)
        dir_binary = proc.dir_threshold(img)

        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        s_binary = proc.compute_white_yellow_lines(hls)

        combined_gradients = np.zeros_like(dir_binary)
        combined_gradients[(gradx == 1) | ((mag_binary == 1) & (dir_binary == 1) & (grady == 1))] = 1

        combined = np.zeros_like(s_binary)
        combined[(combined_gradients == 1) | (s_binary == 1)] = 1

        return combined

    def apply_perspective_transform(self, img, src, dst):
        return cv2.warpPerspective(img, (cv2.getPerspectiveTransform(src, dst)),
                                   (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)

    def fit_polynomial(self, y, x):
        return np.polyfit(y, x, 2)

    def detect_lane_lines_pixels(self, binary_warped):
        # Take a histogram of the bottom half of the image
        histogram = np.sum(binary_warped[binary_warped.shape[0] // 2:, :], axis=0)
        # Create an output image to draw on and visualize the result
        out_img = np.dstack((binary_warped, binary_warped, binary_warped))
        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0] // 2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        # Set height of windows - based on nwindows above and image shape
        window_height = np.int(binary_warped.shape[0] // nwindows)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        total_non_zeros = len(nonzeroy)
        # Current positions to be updated later for each window in nwindows
        leftx_current = leftx_base
        rightx_current = rightx_base

        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []

        left_lane_line = LaneLine()
        right_lane_line = LaneLine()

        # Step through the windows one by one
        if self.previous_left_lane_line is None and self.previous_right_lane_line is None:
            left_lane_inds, right_lane_inds = self.find_pixels_on_lanes(binary_warped, window_height,
                                                                        leftx_current, rightx_current, nonzerox,
                                                                        nonzeroy)

        else:
            # We have already computed the lane lines polynomials from a previous image
            left_lane_inds = ((nonzerox > (self.previous_left_lane_line.polynomial_coeff[0] * (nonzeroy ** 2)
                                           + self.previous_left_lane_line.polynomial_coeff[1] * nonzeroy
                                           + self.previous_left_lane_line.polynomial_coeff[2] - margin))
                              & (nonzerox < (self.previous_left_lane_line.polynomial_coeff[0] * (nonzeroy ** 2)
                                             + self.previous_left_lane_line.polynomial_coeff[1] * nonzeroy
                                             + self.previous_left_lane_line.polynomial_coeff[2] + margin)))

            right_lane_inds = ((nonzerox > (self.previous_right_lane_line.polynomial_coeff[0] * (nonzeroy ** 2)
                                            + self.previous_right_lane_line.polynomial_coeff[1] * nonzeroy
                                            + self.previous_right_lane_line.polynomial_coeff[2] - margin))
                               & (nonzerox < (self.previous_right_lane_line.polynomial_coeff[0] * (nonzeroy ** 2)
                                              + self.previous_right_lane_line.polynomial_coeff[1] * nonzeroy
                                              + self.previous_right_lane_line.polynomial_coeff[2] + margin)))

            non_zero_found_left = np.sum(left_lane_inds)
            non_zero_found_right = np.sum(right_lane_inds)
            non_zero_found_pct = (non_zero_found_left + non_zero_found_right) / total_non_zeros

            if non_zero_found_pct < 0.85:
                left_lane_inds, right_lane_inds = self.find_pixels_on_lanes(binary_warped, window_height, leftx_current,
                                                                            rightx_current, nonzerox,
                                                                            nonzeroy)
                non_zero_found_left = np.sum(left_lane_inds)
                non_zero_found_right = np.sum(right_lane_inds)
                non_zero_found_pct = (non_zero_found_left + non_zero_found_right) / total_non_zeros
                print("[Sliding windows] Found pct={0}".format(non_zero_found_pct))

        # Extract left and right line pixel positions
        left_lane_line.non_zero_x = nonzerox[left_lane_inds]
        left_lane_line.non_zero_y = nonzeroy[left_lane_inds]
        left_fit = self.fit_polynomial(left_lane_line.non_zero_y, left_lane_line.non_zero_x)
        left_lane_line.polynomial_coeff = left_fit

        right_lane_line.non_zero_x = nonzerox[right_lane_inds]
        right_lane_line.non_zero_y = nonzeroy[right_lane_inds]
        right_fit = self.fit_polynomial(right_lane_line.non_zero_y, right_lane_line.non_zero_x)
        right_lane_line.polynomial_coeff = right_fit

        # Generate x and y values for plotting
        ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
        left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

        left_lane_line.line_fit_x = left_fitx
        right_lane_line.line_fit_x = right_fitx

        return left_lane_line, right_lane_line, out_img

    def find_pixels_on_lanes(self, binary_warped, window_height, leftx_current, rightx_current, nonzerox, nonzeroy):
        left_lane_inds = []
        right_lane_inds = []
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = binary_warped.shape[0] - (window + 1) * window_height  # 720 - 80 -- 640
            win_y_high = binary_warped.shape[0] - window * window_height  # 720 - 0 -- 720

            win_xleft_low = leftx_current - margin  # Update this  # base peakLeft - 100
            win_xleft_high = leftx_current + margin  # Update this
            win_xright_low = rightx_current - margin  # Update this
            win_xright_high = rightx_current + margin

            # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                              (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                               (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)

            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))  # imp_ for next image
            if len(good_right_inds) > minpix:
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))  # imp_for next frame

        # Concatenate the arrays of indices (previously was a list of lists of pixels)
        try:
            left_lane_inds = np.concatenate(left_lane_inds)
            right_lane_inds = np.concatenate(right_lane_inds)
        except ValueError:
            # Avoids an error if the above is not implemented fully
            pass

        return left_lane_inds, right_lane_inds

    def measure_curvature(self, left_lane, right_lane):
        ploty = self.ploty
        y_eval = np.max(ploty)

        # Define conversions in x and y from pixels space to meters
        leftx = left_lane.line_fit_x
        rightx = right_lane.line_fit_x

        # Fit new polynomials: find x for y in real-world space
        left_fit_cr = np.polyfit(ploty * self.ym_per_px, leftx * self.xm_per_px, 2)
        right_fit_cr = np.polyfit(ploty * self.ym_per_px, rightx * self.xm_per_px, 2)

        # Now calculate the radii of the curvature
        left_curverad = ((1 + (
                2 * left_fit_cr[0] * y_eval * self.ym_per_px + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
            2 * left_fit_cr[0])
        right_curverad = ((1 + (
                2 * right_fit_cr[0] * y_eval * self.ym_per_px + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
            2 * right_fit_cr[0])

        # Use our computed polynomial to determine the car's center position in image space, then
        left_fit = left_lane.polynomial_coeff
        right_fit = right_lane.polynomial_coeff

        center_offset_img_space = (((left_fit[0] * y_eval ** 2 + left_fit[1] * y_eval + left_fit[2]) +
                                    (right_fit[0] * y_eval ** 2 + right_fit[1] * y_eval + right_fit[
                                        2])) / 2) - lane_center
        center_offset_real_world_m = center_offset_img_space * self.xm_per_px

        # Now our radius of curvature is in meters
        return left_curverad, right_curverad, center_offset_real_world_m

    def draw_lane_area(self, warped_img, undist_img, left_line, right_line):
        """
        Returns an image where the inside of the lane has been colored in bright green
        """
        # Create an image to draw the lines on
        color_warp = warped_img

        ploty = np.linspace(0, warped_img.shape[0] - 1, warped_img.shape[0])
        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([left_line.line_fit_x, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_line.line_fit_x, ploty])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        newwarp = cv2.warpPerspective(color_warp, cv2.getPerspectiveTransform(dst_pts, src_pts),
                                      (undist_img.shape[1], undist_img.shape[0]))
        # Combine the result with the original image
        result = cv2.addWeighted(undist_img, 1, newwarp, 0.3, 0)

        return result

    def draw_lane_curvature_text(self, img, left_curvature_meters, right_curvature_meters, center_offset_meters):
        """
        Returns an image with curvature information inscribed
        """

        offset_y = 100
        offset_x = 100

        template = "{0:17}{1:17}{2:17}"
        txt_header = template.format("Left Curvature", "Right Curvature", "Center Alignment")
        print(txt_header)
        txt_values = template.format("{:.4f}m".format(left_curvature_meters),
                                     "{:.4f}m".format(right_curvature_meters),
                                     "{:.4f}m Right".format(center_offset_meters))
        if center_offset_meters < 0.0:
            txt_values = template.format("{:.4f}m".format(left_curvature_meters),
                                         "{:.4f}m".format(right_curvature_meters),
                                         "{:.4f}m Left".format(math.fabs(center_offset_meters)))

        print(txt_values)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, txt_header, (offset_x, offset_y), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(img, txt_values, (offset_x, offset_y + 50), font, 1, (255, 255, 255), 2, cv2.LINE_AA)

        return img


if __name__ == '__main__':
    img_paths = glob.glob("./testimages_harder_challenge/*.jpg")
    img_paths.sort(key=lambda f: int(re.sub('\D', '', f)))
    classifier = AdvancedLaneLineDetector()
    i = 1
    for img_path in img_paths:
        img_actual = mpimg.imread(img_path)
        output = classifier.process_frame(img_actual)
        plt.imsave("output_part3/" + str(i), output, format="jpg")
        i += 1
