import numpy as np
import math
from skimage.measure import compare_ssim, compare_nrmse
import cv2


class MetricCalculator:

    """
        provides the functionality to calculate different quality metrics
    """

    def __init__(self, video_cap_raw, video_cap_coded, color_space_type):
        self.video_cap_raw = video_cap_raw
        self.video_cap_coded = video_cap_coded
        self.color_space_type = color_space_type
        self.frames = []
        self.dataCollection = []
        self.avgValue = 0
        self.MAX_PIXEL = 255

    def __calc_ssim(self, img1, img2, multi_channel):
        """
            calculates the structural similarity between two images
        :param img1: raw image (original)
        :param img2: coded image
        :return: ssim value

        The following settings are necessary to match the implementation of Wang et. al. [1]

        References:
        --------

        .. [1] Wang, Z., Bovik, A. C., Sheikh, H. R., & Simoncelli, E. P.
           (2004). Image quality assessment: From error visibility to
           structural similarity. IEEE Transactions on Image Processing,
           13, 600-612.
           https://ece.uwaterloo.ca/~z70wang/publications/ssim.pdf,
           :DOI:`10.1109/TIP.2003.819861`
        """

        return compare_ssim(img1, img2, gaussian_weights=True,
                            sigma=1.5, use_sample_covariance=False, multichannel=multi_channel)

    def __calc_psnr(self, img1, img2):
        """
            calculates the peak-signal-to noise ration between two images
        :param img1: original image
        :param img2: coded image
        :return: psnr value
        """
        target_data = np.array(img2, dtype=np.float64)
        ref_data = np.array(img1, dtype=np.float64)

        diff = ref_data - target_data
        diff = diff.flatten('C')

        mse = math.sqrt(np.mean(diff ** 2.))

        # if black frames appear during the measurement (leading to mse=zero), return the maximum float value for them.
        if mse == 0:
            return float("inf")

        return 20 * math.log10(self.MAX_PIXEL / mse)

    def __calc_nrmse(self, img1, img2, norm_type):
        """
            calculate normalized root mean-squared error (NRMSE)

        :param img1: raw image
        :param img2: coded image
        :return: nrmse value
        """
        return compare_nrmse(img1, img2, norm_type)

    def check_selected_metric(self, selected_metric, img1, img2):
        """
            check which metric is selected
        :param selected_metric: metric to calculate
        :param img1: original image
        :param img2: coded image
        :return: metric value
        """
        if selected_metric == "PSNR":

            return self.__calc_psnr(img1=img1, img2=img2)

        elif selected_metric == "SSIM":

            multi = False

            if self.color_space_type == "RGB":
                multi = True

            return self.__calc_ssim(img1=img1, img2=img2, multi_channel=multi)

        elif selected_metric == "NRMSE":

            return self.__calc_nrmse(img1=img1, img2=img2, norm_type='min-max')

        else:
            print("Selected metric not allowed!!!")
            exit(-1)

    def separate_channels(self, raw_frame, coded_frame):
        """
            used to convert color space from RGB to YUV if needed and to separate the channels
        :param raw_frame: original image
        :param coded_frame: coded image
        :return: separated channels for selected color space
        """

        raw_channels = []
        coded_channels = []

        if self.color_space_type == "RGB":
            # extract r,g,b channels and calculate metric for each channel
            raw_channels = cv2.split(raw_frame)
            coded_channels = cv2.split(coded_frame)

        elif self.color_space_type == "YUV":
            raw_yuv = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2YCrCb)
            coded_yuv = cv2.cvtColor(coded_frame, cv2.COLOR_BGR2YCrCb)

            # extract Y [Luminance] channel
            raw_channels = cv2.split(raw_yuv)
            coded_channels = cv2.split(coded_yuv)

        else:
            print("Wrong color space selected!!!")
            exit(-1)

        return raw_channels, coded_channels

    def perform_measuring(self, selected_metric):
        """
            performs / starts the metric measuring
        :param selected_metric: metric to calculate
        :return: Tuple containing two lists (frame numbers and the corresponding metric values)
        """
        metric_val = 0
        frame_number = 1

        while True:

            raw_ret, raw_frame = self.video_cap_raw.read()
            cod_ret, coded_frame = self.video_cap_coded.read()

            # check if the return value for frame capturing is false
            # -> this means that we have reached the end of the video
            if not raw_ret or not cod_ret or cv2.waitKey(1) & 0xFF == ord('q'):
                break

            raw_channels, coded_channels = self.separate_channels(raw_frame, coded_frame)

            # check selected color space
            if self.color_space_type == "RGB":

                b_val = self.check_selected_metric(selected_metric, raw_channels[0], coded_channels[0])
                g_val = self.check_selected_metric(selected_metric, raw_channels[1], coded_channels[1])
                r_val = self.check_selected_metric(selected_metric, raw_channels[2], coded_channels[2])

                metric_val = (b_val + g_val + r_val)/3
                # metric_val = self.check_selected_metric(selected_metric, raw_frame, coded_frame)

            elif self.color_space_type == "YUV":

                # perform psnr for y-channel
                metric_val = self.check_selected_metric(selected_metric, raw_channels[0], coded_channels[0])

                # upsnr = self.check_selected_metric(selected_metric, raw_channels[1], coded_channels[1])
                # vpsnr = self.check_selected_metric(selected_metric, raw_channels[2], coded_channels[2])

            else:
                print("Wrong color space.....")
                exit(-1)

                # if there are black frames inside our test sequences then the psnr for them would be infinity
                # therefore we need to check if there are some infinity values to skip them
            if metric_val == math.inf:
                continue

            # print current value
            print("FRAME {0}: {1}-VALUE [{2}] -> {3:.4f} [dB] ".format(frame_number, selected_metric,
                                                                       self.color_space_type, metric_val))

            # append data to frames and data list
            self.frames.append(frame_number)
            self.dataCollection.append(metric_val)

            # increase frame number and add metric value to avg
            frame_number += 1
            self.avgValue += metric_val

        # after the metric measuring has finished, close video capture
        self.video_cap_raw.release()
        self.video_cap_coded.release()

        # return tuple containing lists with frame numbers and metric measurements
        return self.frames, self.dataCollection

    def get_avg_value(self):
        """
            Return average metric value
        :return: avg value
        """

        # to get avg value divide the collected sum over all measurements by the number of frames
        return self.avgValue / len(self.frames)
