import numpy as np
import math
import argparse
import cv2


class MetricCalculator:

    def __init__(self, video_cap_raw, video_cap_coded, colorspacetype):
        self.video_cap_raw = video_cap_raw
        self.video_cap_coded = video_cap_coded
        self.colorspacetype = colorspacetype
        self.frames = []
        self.dataCollection = []
        self.avgValue = 0

    def __calcssim(self, img1, img2):

        return 0.0

    def __calcpsnr(self, img1, img2):

        target_data = np.array(img1, dtype=np.float64)
        ref_data = np.array(img2, dtype=np.float64)

        diff = ref_data - target_data
        diff = diff.flatten('C')

        mse = math.sqrt(np.mean(diff ** 2.))

        return 20 * math.log10(255 / mse)

    def check_selected_metric(self, selected_metric, img1, img2):

        if selected_metric == "PSNR":
            return self.__calcpsnr(img1=img1, img2=img2)
        elif selected_metric == "SSIM":
            return self.__calcssim(img1=img1, img2=img2)
        else:
            print("Selected metric not allowed!!!")
            exit(-1)

    def convert_to_yuv(self, raw_frame, coded_frame):

        raw_yuv = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2YCrCb)
        coded_yuv = cv2.cvtColor(coded_frame, cv2.COLOR_BGR2YCrCb)

        # extract Y [Luminance] channel
        y_raw, _, _ = cv2.split(raw_yuv)
        y_coded, _, _ = cv2.split(coded_yuv)

        # cv2.imshow("Y-Raw", y_raw)
        # cv2.imshow("Y-Coded", y_coded)

        return y_raw, y_coded

    def perform_measuring(self, selected_metric):

        metric_val = 0
        frame_number = 1

        while True:

            raw_ret, raw_frame = self.video_cap_raw.read()
            cod_ret, coded_frame = self.video_cap_coded.read()

            # check if the return value for frame capturing is false
            # -> this means that we have reached the end of the video
            if not raw_ret or not cod_ret or frame_number == 50 or cv2.waitKey(1) & 0xFF == ord('q'):
                break

                # check selected color space
            if self.colorspacetype == "RGB":

                metric_val = self.check_selected_metric(selected_metric, raw_frame, coded_frame)

                print("FRAME {0}".format(frame_number))
                print("{0}-VALUE [RGB] -> {1:.2f} [dB] ".format(selected_metric, metric_val))

            elif self.colorspacetype == "YUV":

                y_raw, y_coded = self.convert_to_yuv(raw_frame, coded_frame)

                # perform psnr for y-channel
                metric_val = self.check_selected_metric(selected_metric, y_raw, y_coded)
                # upsnr = psnr(u_raw, u_coded)
                # vpsnr = psnr(v_raw, v_coded)

                # print current value
                print("FRAME {0}".format(frame_number))
                print("{0}-VALUE [Y-Channel] -> {1:.2f} [dB] ".format(selected_metric, metric_val))

            else:
                print("Wrong color space.....")
                exit(-1)

                # if there are black frames inside our test sequences then the psnr for them would be infinity
                # therefore we need to check if there are some infinity values to skip them
            if metric_val == math.inf:
                continue

            # append data to frames and data list
            self.frames.append(frame_number)
            self.dataCollection.append(metric_val)

            frame_number += 1
            self.avgValue += metric_val

        self.video_cap_raw.release()
        self.video_cap_coded.release()

        return self.frames, self.dataCollection

    def get_avg_value(self):
        return self.avgValue / len(self.frames)
