import math
import os
import sys
from collections import deque
from multiprocessing.pool import Pool
from time import time

import cv2 as cv
from cv2 import split, CAP_PROP_FRAME_COUNT

from utils.common_metrics import calc_psnr, calc_ssim, calc_wpsnr, calc_ws_psnr, calc_vpsnr
from utils.head_motion_parser import process_log
from utils.parse_video_files import get_video_files
# variables defining maximal pixel value and progressbar length
from utils.vector_util import Viewport


def get_metric(selected_metric):
    """
           used to check which metric function to execute
       :param selected_metric: currently selected metric
       :return: function representing the selected metric
    """

    switcher = {
        "PSNR": calc_psnr,
        "WS-PSNR": calc_ws_psnr,
        "SSIM": calc_ssim,
        "W-PSNR": calc_wpsnr
        # "V-PSNR": calc_vpsnr
    }
    # get the selected metric to calculate
    m = switcher.get(selected_metric, calc_psnr)
    return m


class MetricCalculator:
    def __init__(self,
                 num_processes,
                 num_frames_skip,
                 raw_file_path,
                 coded_file_path,
                 color_space,
                 metric):
        self.num_processes = num_processes
        self.num_frames_skip = num_frames_skip
        self.raw_file_path = raw_file_path
        self.coded_file_path = coded_file_path
        self.color_space = color_space
        self.metric = metric
        self.MAX_PIXEL = 255
        self.bar_len = 60

    def start_processing(self):
        self.__perform_processing()

    def __perform_processing(self) -> None:
        """
            perform the metric calculation

        :param num_processes: number of available processes
        :param num_frames_skip: number of frames to skip
        :param raw_file_path: raw video file path
        :param coded_files_path: coded video file path
        :param metric: selected metric

        """
        # define metric which would be calculated
        _metric_func = self.__get_metric(self.metric)

        # high performance object used to cache async tasks
        _task_buffer = deque()

        # start calculation timer
        start_time = time()

        # open a pool of processes used to calculate the selected metric
        with Pool(processes=self.num_processes) as _pool:

            # parse encoded video files from given directory
            encoded_files = get_video_files(self.coded_file_path)

            # if current selected metric is v-psnr then we need to initialize some configurations
            if self.metric == "V-PSNR":
                _mvmt_file_path = input("Enter log file path: ")
                _fps = float(input("Enter fps: "))
                _num_frames = int(input("Enter number of frames: "))

                _mvmt_data = process_log(_mvmt_file_path, _fps, _num_frames)

                _width = int(input("Enter viewport width: "))
                _height = int(input("Enter viewport height: "))
                _fovx = float(input("Enter field of view (x-axis): "))
                _vp = Viewport(_width, _height, _fovx)
                print(_mvmt_data[0])

            # perform calculation for each given encoded file
            for encoded_file in encoded_files:

                # init necessary stuff

                _frame_count = 1
                _avg_value = 0.0

                # set current raw video capture
                _cap_raw = cv.VideoCapture(self.raw_file_path)

                # set current encoded video capture
                _cap_coded = cv.VideoCapture(encoded_file)

                # set number of frames to skip
                if self.num_frames_skip > 1:
                    _cap_raw.set(cv.CAP_PROP_POS_FRAMES, self.num_frames_skip)
                    _cap_coded.set(cv.CAP_PROP_POS_FRAMES, self.num_frames_skip)

                # get number of frames to process
                num_frames = _cap_raw.get(CAP_PROP_FRAME_COUNT)

                print(f"Number of frames to process -> {num_frames}")
                # cut out the video name from the given video path
                _, _coded_file_basename = os.path.split(encoded_file)
                print(f"Selected coded video file -> {_coded_file_basename}\n")

                # print progressbar to console -> 0.0%
                self.__print_progress(0, num_frames)

                _log_count = 0

                # start the calculation
                while True:
                    # process generated tasks
                    while len(_task_buffer) > 0 and _task_buffer[0].ready():
                        self.__print_progress(_frame_count, num_frames)

                        # pop element from rightmost side
                        value = _task_buffer.pop().get()
                        # skip black frames returning math.inf as metric value
                        if value == math.inf:
                            continue

                        # print current calculation
                        # print("PSNR Value     :  %.3f [dB]" % value)

                        # add current value to avg and increase frame count
                        _avg_value += value
                        _frame_count += 1

                    # if length of dequeue is smaller than number of available processes -> start generating new tasks
                    if len(_task_buffer) < self.num_processes:
                        # read frames
                        has_raw_frames, raw_frame = _cap_raw.read()
                        has_coded_frames, coded_frame = _cap_coded.read()

                        # check if end of video is reached
                        if not has_raw_frames or not has_coded_frames:
                            _task_buffer.clear()
                            break

                        # check whether the raw and coded videos are of same shape
                        # otherwise continue to next encoded file
                        elif raw_frame.shape != coded_frame.shape:
                            print("video shape doesn't match...")
                            break

                        # check whether YUV or RGB, etc. color space is selected
                        if self.color_space == "YUV":

                            # _raw_channels, _coded_channels = separate_channels(raw_frame, coded_frame,
                            # _colorSpaceType) generate new asynchronous task
                            _yuv_raw = cv.cvtColor(raw_frame, cv.COLOR_BGR2YCrCb)
                            _yuv_coded = cv.cvtColor(coded_frame, cv.COLOR_BGR2YCrCb)

                            # check which metric is selected
                            if self.metric in {"PSNR", "WS-PSNR", "SSIM"}:
                                task = _pool.apply_async(_metric_func, (split(_yuv_raw)[0], split(_yuv_coded)[0]))

                            elif self.metric == "V-PSNR":
                                vd = _mvmt_data[_log_count]
                                # print(f"x -> {vd.x} | y -> {vd.y} | z -> {vd.z}")
                                task = _pool.apply_async(_metric_func,
                                                         (split(_yuv_raw)[0], split(_yuv_coded)[0], vd, _vp))
                                _log_count += 1
                            else:
                                task = _pool.apply_async(_metric_func, (split(_yuv_raw)[0], split(_yuv_coded)[0]))

                        else:
                            # if selected color space is RGB, etc. -> then calculate the metric using all 3 channels
                            # combined
                            if self.metric == "SSIM":
                                task = _pool.apply_async(_metric_func, (raw_frame, coded_frame, True))
                            else:
                                task = _pool.apply_async(_metric_func, (raw_frame, coded_frame))

                        # append task to left side of queue
                        _task_buffer.appendleft(task)

                # release current video capture
                _cap_raw.release()
                _cap_coded.release()

                print('calculation finished\n')
                # print average metric value
                print(f"average {self.metric} value    :  {_avg_value / _frame_count}\n")

            # show duration of processing
            print(f"duration of measuring    : {time() - start_time} s")

    @staticmethod
    def __get_metric(selected_metric):
        """
               used to check which metric function to execute
           :param selected_metric: currently selected metric
           :return: function representing the selected metric
        """

        switcher = {
            "PSNR": calc_psnr,
            "WS-PSNR": calc_ws_psnr,
            "SSIM": calc_ssim,
            "W-PSNR": calc_wpsnr,
            "V-PSNR": calc_vpsnr
        }
        # get the selected metric to calculate
        m = switcher.get(selected_metric, calc_psnr)
        return m

    def __print_progress(self, iteration, total):
        """
            shows a progress bar during video processing
        :param iteration:
        :param total:
        :return:
        """
        filled_len = int(round(self.bar_len * iteration / float(total)))

        percents = round(100.0 * iteration / float(total), 1)
        bar = '=' * filled_len + '-' * (self.bar_len - filled_len)

        sys.stdout.write('\r[%s] %s%s ...%s' % (bar, percents, '%', "Processing Progress"))
        sys.stdout.flush()
