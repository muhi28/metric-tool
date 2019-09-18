
import argparse
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


def __calc_ws_psnr(img1, img2, _t):
    """
        performs the weighted spherical psnr calculation
    :param img1: original frame
    :param img2: coded/reference/reconstructed frame
    :param _t: frame time
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

    return _sum_val, _t


def __calc_psnr(img1, img2, _t):
    """
        calculates the peak-signal-to noise ration between two images
    :param img1: original image
    :param img2: coded image
    :param _t: frame time
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

    return 10 * log10((MAX_PIXEL ** 2) / mse), _t


def __init_argparser():
    """
        used to generate and initialize the argument parser
    :return: dict containing the available arguments
    """
    # create arg parser
    arg_parser = argparse.ArgumentParser()

    # add argument elements to argparse
    arg_parser.add_argument("-r", "--raw", required=True, help="original input video")
    arg_parser.add_argument("-e", "--encoded", required=True, help="encoded input video")
    arg_parser.add_argument("-c", "--colorspace", required=True, help="color space in which to perform measurements")
    arg_parser.add_argument("-m", "--metric", required=True, help="metric to measure")

    # parse all arguments
    return vars(arg_parser.parse_args())


def __check_video_resolutions(raw_cap, coded_cap):
    """
        check resolution of of video to compare

    :param raw_cap: raw video capture
    :param coded_cap: coded video capture
    :return:
             True -> same resolution
             False -> resolution not the same
    """

    raw_width = int(raw_cap.get(CAP_PROP_FRAME_WIDTH))
    raw_height = int(raw_cap.get(CAP_PROP_FRAME_HEIGHT))
    coded_width = int(coded_cap.get(CAP_PROP_FRAME_WIDTH))
    coded_height = int(coded_cap.get(CAP_PROP_FRAME_HEIGHT))

    # at the same time we need to init the frame width and height four our common metrics
    return (raw_width == coded_width) and (raw_height == coded_height)


def __get_metric(selected_metric):
    """
        used to check which metric function to execute
    :param selected_metric: currently selected metric
    :return: function representing the selected metric
    """
    switcher = {
        "PSNR": __calc_psnr,
        "WS-PSNR": __calc_ws_psnr
    }

    # get the selected metric to calculate
    m = switcher.get(selected_metric, lambda: "PSNR")

    return m


if __name__ == '__main__':

    if len(sys.argv) <= 1:
        print("Update the values -> raw video file/image and coded video file / image")
        exit(0)

    # parse all arguments
    _args = __init_argparser()

    # extract parameters
    _rawFilePath = _args["raw"]  # get raw file path
    _codedFilePath = _args["encoded"]  # get coded file path
    _colorSpaceType = _args["colorspace"]  # get selected color space to calculate the metrics
    _metricToCalculate = _args["metric"]  # get selected metric to calculate

    _cap_raw = cv.VideoCapture(_rawFilePath)
    _cap_coded = cv.VideoCapture(_codedFilePath)

    # parse basename
    _, _raw_file_basename = os.path.split(_rawFilePath)
    _, _coded_file_basename = os.path.split(_codedFilePath)

    _num_threads = int(cv.getNumberOfCPUs() / 2) + 1
    # _pool = Pool(processes=_num_threads)
    _pending = deque()

    _latency = _StatValue()
    _frame_interval = _StatValue()
    _last_frame_time = _clock()
    _frame_count = 1
    _avg_value = 0

    print("Start calculation ....\n")

    print("Settings:")
    print("number of threads    :  {0}".format(_num_threads))
    print("color space -> {0}".format(_colorSpaceType))
    print("Selected raw video file -> {0}".format(_raw_file_basename))
    print("Selected coded video file -> {0}".format(_coded_file_basename))
    print("FPS -> {0}".format(_cap_raw.get(CAP_PROP_FPS)))
    print("Color Space -> {0}".format(_colorSpaceType))
    print("Selected Metric -> {0}\n".format(_metricToCalculate))

    # define metric which would be calculated
    _metric_func = __get_metric(_metricToCalculate)

    start_time = time()

    with Pool(processes=_num_threads) as _pool:
        # start the calculation
        while True:

            # process generated tasks
            while len(_pending) > 0 and _pending[0].ready():
                # pop element from rightmost side
                value, frame_time = _pending.pop().get()

                # update latency
                _latency.update(_clock() - frame_time)

                # print current calculation
                print("latency        :  %.1f ms" % (_latency.value * 1000))
                print("frame interval :  %.1f ms" % (_frame_interval.value * 1000))
                print("PSNR Value     :  %.3f [dB]" % value)
                print("Frame count    :  {0}\n".format(_frame_count))

                # add current value to avg and increase frame count
                _avg_value += value
                _frame_count += 1

            # if length of dequeue is smaller than number of available threads -> start generating new tasks
            if len(_pending) < _num_threads:
                # read frames
                has_raw_frames, raw_frame = _cap_raw.read()
                has_coded_frames, coded_frame = _cap_coded.read()

                if not has_raw_frames or not has_coded_frames:
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

                    task = _pool.apply_async(_metric_func, (split(_yuv_raw)[0], split(_yuv_coded)[0], t))
                else:
                    # if selected color space is RGB, etc. -> then calculate the metric using all 3 channels combined
                    task = _pool.apply_async(_metric_func, (raw_frame, coded_frame, t))

                # append task to left side of queue
                _pending.appendleft(task)

    print('calculation finished\n')

    print("duration of measureing    : {0} ms".format((time() - start_time)))
    print("average {0} value    :  {1}".format(_metricToCalculate, _avg_value / _frame_count))
