import numpy as np
import cv2


class PreprocessingPipeline:
    def __init__(self, kernel_size, sob_thresh, mag_thresh, dir_thresh):
        self.kernel_size = kernel_size
        self.sobel_thresh = sob_thresh
        self.mag_thresh = mag_thresh
        self.dir_thresh = dir_thresh

    def abs_sobel_thresh(self, img, orient='x'):
        # 1) Convert to grayscale
        gimg = (cv2.cvtColor(img, cv2.COLOR_RGB2LAB))[:, :, 0]
        # 2) Take the derivative in x or y given orient = 'x' or 'y'
        sobel_img = (cv2.Sobel(gimg, cv2.CV_64F, 1, 0, self.kernel_size) if orient == 'x' else
                     cv2.Sobel(gimg, cv2.CV_64F, 0, 1, self.kernel_size))
        # 3) Take the absolute value of the derivative or gradient
        abs_sobel = np.absolute(sobel_img)
        # 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
        scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
        # 5) Create a mask of 1's where the scaled gradient magnitude
        # is > thresh_min and < thresh_max
        mask_sobel = np.zeros_like(scaled_sobel)
        mask_sobel[(scaled_sobel >= self.sobel_thresh[0]) & (scaled_sobel <= self.sobel_thresh[1])] = 1

        return mask_sobel

    def mag_threshold(self, img):
        # Apply the following steps to img
        # 1) Convert to grayscale
        gray = (cv2.cvtColor(img, cv2.COLOR_RGB2LAB))[:, :, 0]
        # 2) Take the gradient in x and y separately
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=self.kernel_size)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=self.kernel_size)
        # 3) Calculate the magnitude
        s_ss = ((sobel_x ** 2) + (sobel_y ** 2))  # reduced time of execution than np.square()
        mag_sobel = np.sqrt(s_ss)
        scale_factor = np.max(mag_sobel) / 255
        # 4) Scale to 8-bit (0 - 255) and convert to type = np.uint8
        bin_img = np.uint8(mag_sobel / scale_factor)
        # 5) Create a binary mask where mag thresholds are met
        sbinary = np.zeros_like(bin_img)
        sbinary[(bin_img >= self.mag_thresh[0]) & (bin_img <= self.mag_thresh[1])] = 1
        return sbinary

    def dir_threshold(self, img):
        # Apply the following steps to img
        # 1) Convert to grayscale
        gray = (cv2.cvtColor(img, cv2.COLOR_RGB2LAB))[:, :, 0]
        # 2) Take the gradient in x and y separately
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=self.kernel_size)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=self.kernel_size)
        # 3) Take the absolute value of the x and y gradients
        abs_x = np.absolute(sobel_x)
        abs_y = np.absolute(sobel_y)
        # 4) Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient
        bin_img = np.arctan2(abs_y, abs_x)
        # 5) Create a binary mask where direction thresholds are met
        bin_out = np.zeros_like(bin_img)
        bin_out[(bin_img >= self.dir_thresh[0]) & (bin_img <= self.dir_thresh[1])] = 1
        # 6) Return this mask as your binary_output image
        binary_output = np.copy(bin_out)  # Remove this line
        return binary_output

    def compute_white_yellow_lines(self, rgb_img,
                                       H_yell_min=15, H_yell_max=35,
                                       L_yell_min=30, L_yell_max=204, S_yell_min=115, S_yell_max=255, H_whit_min=0,
                                       H_whit_max=255,
                                       L_whit_min=200, L_whit_max=255, S_whit_min=0, S_whit_max=255):
        """
        Returns a binary thresholded image containing the white and yellow
        contents of the image which is mostly targeted for the lanes
        """
        hls_img = rgb_img

        # Compute a binary thresholded image where yellow is isolated from HLS components
        img_hls_yellow_bin = np.zeros_like(hls_img[:, :, 0])
        img_hls_yellow_bin[((hls_img[:, :, 0] >= H_yell_min) & (hls_img[:, :, 0] <= H_yell_max))
                           & ((hls_img[:, :, 1] >= L_yell_min) & (hls_img[:, :, 1] <= L_yell_max))
                           & ((hls_img[:, :, 2] >= S_yell_min) & (hls_img[:, :, 2] <= S_yell_max))
                           ] = 1

        # Compute a binary thresholded image where white is isolated from HLS components
        img_hls_white_bin = np.zeros_like(hls_img[:, :, 0])
        img_hls_white_bin[((hls_img[:, :, 0] >= H_whit_min) & (hls_img[:, :, 0] <= H_whit_max))
                          & ((hls_img[:, :, 1] >= L_whit_min) & (hls_img[:, :, 1] <= L_whit_max))
                          & ((hls_img[:, :, 2] >= S_whit_min) & (hls_img[:, :, 2] <= S_whit_max))
                          ] = 1

        # Now combine both
        img_hls_white_yellow_bin = np.zeros_like(hls_img[:, :, 0])
        img_hls_white_yellow_bin[(img_hls_yellow_bin == 1) | (img_hls_white_bin == 1)] = 1

        return img_hls_white_yellow_bin
