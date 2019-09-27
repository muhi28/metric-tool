from math import inf

from utils.utilities import separate_channels
from cv2 import waitKey, CAP_PROP_FRAME_WIDTH, CAP_PROP_FRAME_HEIGHT
from utils.common_metrics import init_frame_data, calc_psnr, calc_ws_psnr, calc_ssim, calc_nrmse


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
        init_frame_data(self.frame_height, self.frame_width)

    def calc_selected_metric(self, selected_metric, img_tuples):
        """
            check which metric is selected
        :param img_tuples: containing both raw and coded image
        :param selected_metric: metric to calculate
        :return: metric value
        """
        if selected_metric in {"PSNR", "WPSNR"}:

            return calc_psnr(img1=img_tuples[0], img2=img_tuples[1])

        elif selected_metric == "WS-PSNR":

            return calc_ws_psnr(img1=img_tuples[0], img2=img_tuples[1])

        elif selected_metric == "SSIM":

            return calc_ssim(img1=img_tuples[0], img2=img_tuples[1],
                             multi_channel=(True if self.color_space_type in {"RGB", "HVS"} else False))

        elif selected_metric == "NRMSE":

            return calc_nrmse(img1=img_tuples[0], img2=img_tuples[1], norm_type='min-max')

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
