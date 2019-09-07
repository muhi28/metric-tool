import numpy as np
from math import log10, inf, cos, pi
from skimage.measure import compare_ssim, compare_nrmse
from utils.utilities import separate_channels
from cv2 import waitKey, CAP_PROP_FRAME_WIDTH, CAP_PROP_FRAME_HEIGHT


class MetricCalculator:
    """
        provides the functionality to calculate different quality metrics
    """

    def __init__(self, video_cap_raw, video_cap_coded, color_space_type):
        self.video_cap_raw = video_cap_raw
        self.video_cap_coded = video_cap_coded
        self.color_space_type = color_space_type
        self.frame_width = int(video_cap_raw.get(CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(video_cap_raw.get(CAP_PROP_FRAME_HEIGHT))
        self.avgValue = 0
        self.num_frames = 1
        self.MAX_PIXEL = 255

    def __calc_ssim(self, img1, img2, multi_channel):
        """
            calculates the structural similarity between two images
        :rtype: float
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

        mse = np.sum(diff ** 2) / (self.frame_width * self.frame_height)

        # if black frames appear during the measurement (leading to mse=zero), return the maximum float value for them.
        if mse == 0:
            return inf

        return 10 * log10((self.MAX_PIXEL ** 2) / mse)

    def __calc_nrmse(self, img1, img2, norm_type):
        """
            calculate normalized root mean-squared error (NRMSE)

        :param img1: raw image
        :param img2: coded image
        :return: nrmse value
        """
        return compare_nrmse(img1, img2, norm_type)

    def __calc_ws_psnr(self, img1, img2):

        ref_vals = np.array(img1, dtype=np.float64)
        target_vals = np.array(img2, dtype=np.float64)

        sum_val = w_sum = 0.0

        diff = ref_vals - target_vals
        diff = np.abs(diff) ** 2

        pixel_weights = [cos((j - (self.frame_height / 2) + 0.5) * (pi / self.frame_height))
                         for j in range(self.frame_height)]

        for j in range((int(self.frame_height) - 1)):
            for i in range((int(self.frame_width) - 1)):
                sum_val += diff[j, i] * pixel_weights[j]
                w_sum += pixel_weights[j]

        sum_val = sum_val / w_sum

        if sum_val == 0:
            sum_val = 100
        else:
            sum_val = 10 * log10((self.MAX_PIXEL * self.MAX_PIXEL)/sum_val)

        return sum_val

    def __calc_weight(self):
        w_map = []
       # for j in range(self.frame_height):

        #    for i in range(self.frame_width):
         #       val = cos((j - (self.frame_height / 2) + 0.5) * (pi / self.frame_height))
          #      w_map.append(val)

        return w_map

    def calc_selected_metric(self, selected_metric, img_tuples):
        """
            check which metric is selected
        :param img_tuples: containing both raw and coded image
        :param selected_metric: metric to calculate
        :return: metric value
        """
        if selected_metric in {"PSNR", "WPSNR"}:

            return self.__calc_psnr(img1=img_tuples[0], img2=img_tuples[1])

        elif selected_metric == "WS-PSNR":

            return self.__calc_ws_psnr(img1=img_tuples[0], img2=img_tuples[1])

        elif selected_metric == "SSIM":

            return self.__calc_ssim(img1=img_tuples[0], img2=img_tuples[1],
                                    multi_channel=(True if self.color_space_type in {"RGB", "HVS"} else False))

        elif selected_metric == "NRMSE":

            return self.__calc_nrmse(img1=img_tuples[0], img2=img_tuples[1], norm_type='min-max')

        else:
            print("Selected metric not allowed!!!")
            exit(-1)

    def perform_measuring(self, selected_metric):
        """
            performs / starts the metric measuring
        :param selected_metric: metric to calculate
        :return: Tuple containing two lists (frame numbers and the corresponding metric values)
        """
        metric_val = 0
        frames = []
        data_collection = []

        while True:

            raw_ret, raw_frame = self.video_cap_raw.read()
            cod_ret, coded_frame = self.video_cap_coded.read()

            # check if the return value for frame capturing is false
            # -> this means that we have reached the end of the video
            if not raw_ret or not cod_ret or waitKey(1) & 0xFF == ord('q'):
                break

            raw_channels, coded_channels = separate_channels(raw_frame, coded_frame, self.color_space_type)

            if self.color_space_type == "YUV":

                # perform psnr for y-channel
                metric_val = self.calc_selected_metric(selected_metric, (raw_channels[0], coded_channels[0]))

                # check if weighted psnr is selected
                if selected_metric == "WPSNR":
                    u_psnr = self.calc_selected_metric(selected_metric, (raw_channels[1], coded_channels[1]))
                    v_psnr = self.calc_selected_metric(selected_metric, (raw_channels[2], coded_channels[2]))

                    metric_val = ((6 * metric_val) + u_psnr + v_psnr) / 8.0

                # check selected color space
            elif self.color_space_type in {"RGB", "HVS"}:

                val1 = self.calc_selected_metric(selected_metric, (raw_channels[0], coded_channels[0]))
                val2 = self.calc_selected_metric(selected_metric, (raw_channels[1], coded_channels[1]))
                val3 = self.calc_selected_metric(selected_metric, (raw_channels[2], coded_channels[2]))

                metric_val = (val1 + val2 + val3) / 3

            else:
                print("Wrong color space.....")
                exit(-1)

                # if there are black frames inside our test sequences then the psnr for them would be infinity
                # therefore we need to check if there are some infinity values to skip them
            if metric_val == inf:
                continue

            # print current value
            print("FRAME {0}: {1}-VALUE [{2}] -> {3:.3f} [dB] ".format(self.num_frames, selected_metric,
                                                                       self.color_space_type, metric_val))

            # append data to frames and data list
            frames.append(self.num_frames)
            data_collection.append(metric_val)

            # increase frame number and add metric value to avg
            self.num_frames += 1
            self.avgValue += metric_val

        # after the metric measuring has finished, close video capture
        self.video_cap_raw.release()
        self.video_cap_coded.release()

        # return tuple containing lists with frame numbers and metric measurements
        return frames, data_collection

    def get_avg_value(self):
        """
            Return average metric value
        :return: avg value
        """

        # to get avg value divide the collected sum over all measurements by the number of frames
        return self.avgValue / self.num_frames
