import argparse
import gc
import os
import sys
from time import time

import cv2 as cv
import numpy as np

from cv2 import CAP_PROP_FRAME_WIDTH, CAP_PROP_FRAME_HEIGHT, CAP_PROP_FPS, split
from multiprocessing.pool import Pool, ThreadPool
from collections import deque
from utils.utilities import separate_channels
from math import log10, cos, pi, inf

num_frames = 0
avg_value = 0
MAX_PIXEL = 255


class _StatValue:
    def __init__(self, smooth_coef=0.5):
        self.value = None
        self.smooth_coef = smooth_coef

    def update(self, v):
        if self.value is None:
            self.value = v
        else:
            c = self.smooth_coef
            self.value = c * self.value + (1.0 - c) * v


def _clock():
    return cv.getTickCount() / cv.getTickFrequency()


def __calc_ssim(img1, img2, t):
    pass


def __calc_ws_psnr(img1, img2, t):
    """
        performs the weighted spherical psnr calculation
    :param img1: original frame
    :param img2: coded/reference/reconstructed frame
    :param t: frame time
    :return: ws-psnr value
    """
    height, width = img1.shape[0], img1.shape[1]

    _ref_vals = np.array(img1, dtype=np.float64)
    _target_vals = np.array(img2, dtype=np.float64)

    _sum_val = _w_sum = 0.0

    _diff = _ref_vals - _target_vals
    _diff = np.abs(_diff) ** 2
    _diff = _diff.flatten('C')

    _pixel_weights = [cos((j - (height / 2) + 0.5) * (pi / height))
                      for j in range(height)]

    counter = 0
    weight_id = 0

    for val in _diff:

        _sum_val += val * _pixel_weights[weight_id]
        _w_sum += _pixel_weights[weight_id]

        if counter == width:
            counter = 0
            weight_id += 1

        counter += 1

    _sum_val = _sum_val / _w_sum

    if _sum_val == 0:
        _sum_val = 100
    else:
        _sum_val = 10 * log10((MAX_PIXEL * MAX_PIXEL) / _sum_val)

    return _sum_val, t


def __calc_psnr(img1, img2, t):
    """
        calculates the peak-signal-to noise ration between two images
    :param img1: original image
    :param img2: coded image
    :param t: frame time
    :return: psnr value
    """

    dims = img1.shape

    target_data = np.array(img2, dtype=np.float64)
    ref_data = np.array(img1, dtype=np.float64)

    diff = ref_data - target_data
    diff = diff.flatten('C')

    mse = np.sum(diff ** 2) / (dims[1] * dims[0])

    # if black frames appear during the measurement (leading to mse=zero), return the maximum float value for them.
    if mse == 0:
        return inf

    return 10 * log10((MAX_PIXEL ** 2) / mse), t


def __calc_wpsnr(img1, img2, t):
    y_raw, u_raw, v_raw = split(img1)
    y_coded, u_coded, v_coded = split(img2)

    y_psnr, _ = __calc_psnr(y_raw, y_coded, t)
    u_psnr, _ = __calc_psnr(u_raw, u_coded, t)
    v_psnr, _ = __calc_psnr(v_raw, v_coded, t)

    return ((6 * y_psnr) + u_psnr + v_psnr) / 8.0, t


def __init_argparser():
    """
        used to generate and initialize the argument parser
    :return: dict containing the available arguments
    """
    # create arg parser
    arg_parser = argparse.ArgumentParser()

    # add argument elements to argparse
    arg_parser.add_argument("-r", "--raw", required=True, help="original input video")
    arg_parser.add_argument("-e", "--encoded", nargs="+", required=True, help="encoded input video")
    arg_parser.add_argument("-c", "--colorspace", required=True, help="color space in which to perform measurements")
    arg_parser.add_argument("-m", "--metric", required=True, help="metric to measure")

    # parse all arguments
    return vars(arg_parser.parse_args())


def __get_metric(selected_metric):
    """
        used to check which metric function to execute
    :param selected_metric: currently selected metric
    :return: function representing the selected metric
    """
    switcher = {
        "PSNR": __calc_psnr,
        "WS-PSNR": __calc_ws_psnr,
        "SSIM": __calc_ssim,
        "W-PSNR": __calc_wpsnr
    }

    # get the selected metric to calculate
    m = switcher.get(selected_metric, lambda: "PSNR")

    return m


def perform_processing(num_processes, raw_file_path, coded_file_paths, metric) -> None:
    """
        perform the metric calculation

    :param num_processes: number of available processes
    :param raw_file_path: raw video file path
    :param coded_file_paths: coded video file path
    :param metric: selected metric

    """
    # define metric which would be calculated
    _metric_func = __get_metric(_metricToCalculate)

    # initialize time data
    _latency = _StatValue()
    _frame_interval = _StatValue()
    _last_frame_time = _clock()

    # high performance object used to cache async tasks
    _task_buffer = deque()

    # start calculation timer
    start_time = time()

    # open a pool of processes used to calculate the selected metric
    with Pool(processes=_num_processes) as _pool:

        # perform calculation for each given encoded file
        for encoded_file in coded_file_paths:

            # init necessary stuff

            _frame_count = 1
            _avg_value = 0.0

            # set current raw video capture
            _cap_raw = cv.VideoCapture(raw_file_path)

            # set current encoded video capture
            _cap_coded = cv.VideoCapture(encoded_file)

            # cut out the video name from the given video path
            _, _coded_file_basename = os.path.split(encoded_file)
            print("Selected coded video file -> {0}\n".format(_coded_file_basename))

            # start the calculation
            while True:

                # process generated tasks
                while len(_task_buffer) > 0 and _task_buffer[0].ready():
                    # pop element from rightmost side
                    value, frame_time = _task_buffer.pop().get()

                    # update latency
                    _latency.update(_clock() - frame_time)

                    # print current calculation
                    print("Frame Interval :  %.1f ms" % (_frame_interval.value * 1000))
                    print("PSNR Value     :  %.3f [dB]" % value)
                    print("Frame Count    :  {0}\n".format(_frame_count))

                    # add current value to avg and increase frame count
                    _avg_value += value
                    _frame_count += 1

                # if length of dequeue is smaller than number of available threads -> start generating new tasks
                if len(_task_buffer) < num_processes:
                    # read frames
                    has_raw_frames, raw_frame = _cap_raw.read()
                    has_coded_frames, coded_frame = _cap_coded.read()

                    # check if end of video is reached
                    if not has_raw_frames or not has_coded_frames:
                        _task_buffer.clear()
                        print("metric calculation finished....")
                        break

                    # check whether the raw and coded videos are of same shape
                    # otherwise continue to next encoded file
                    if raw_frame.shape != coded_frame.shape:
                        print("video shape doesn't match...")
                        break

                    # update frame interval and latency
                    t = _clock()
                    _frame_interval.update(t - _last_frame_time)
                    _last_frame_time = t

                    # check whether YUV or RGB, etc. color space is selected
                    if _colorSpaceType == "YUV":

                        # _raw_channels, _coded_channels = separate_channels(raw_frame, coded_frame, _colorSpaceType)
                        # generate new asynchronous task
                        _yuv_raw = cv.cvtColor(raw_frame, cv.COLOR_BGR2YCrCb)
                        _yuv_coded = cv.cvtColor(coded_frame, cv.COLOR_BGR2YCrCb)

                        # check which metric is selected
                        if metric in {"PSNR", "WS-PSNR"}:
                            task = _pool.apply_async(_metric_func, (split(_yuv_raw)[0], split(_yuv_coded)[0], t))
                        else:
                            task = _pool.apply_async(_metric_func, (_yuv_raw, _yuv_coded, t))

                    else:
                        # if selected color space is RGB, etc. -> then calculate the metric using all 3 channels
                        # combined
                        task = _pool.apply_async(_metric_func, (raw_frame, coded_frame, t))

                    # append task to left side of queue
                    _task_buffer.appendleft(task)

            # release current video capture
            _cap_raw.release()
            _cap_coded.release()

            print('calculation finished\n')

            # print duration of measuring and average metric value
            print("duration of measuring    : {0} ms".format((time() - start_time)))
            print("average {0} value    :  {1}\n".format(metric, _avg_value / _frame_count))


if __name__ == '__main__':

    if len(sys.argv) <= 1:
        print("Update the values -> raw video file/image and coded video file / image")
        exit(0)

    # parse all arguments
    _args = __init_argparser()

    # extract parameters
    _rawFilePath = _args["raw"]  # get raw file path
    _codedFilesPath = _args["encoded"]  # get coded file path
    _colorSpaceType = _args["colorspace"]  # get selected color space to calculate the metrics
    _metricToCalculate = _args["metric"]  # get selected metric to calculate

    # parse raw basename
    _, _raw_file_basename = os.path.split(_rawFilePath)

    # set number of processes
    _num_processes = int(cv.getNumberOfCPUs())

    print("Start calculation ....\n")

    print("Settings:")
    print("Number of processes    :  {0}".format(_num_processes))
    print("Color Space -> {0}".format(_colorSpaceType))
    print("Selected Metric -> {0}".format(_metricToCalculate))
    print("Selected raw video file -> {0}".format(_raw_file_basename))

    # start the video processing part
    perform_processing(_num_processes, _rawFilePath, _codedFilesPath, _metricToCalculate)
