from __future__ import print_function

import argparse
import os
import sys
import numpy as np
import cv2 as cv

from math import log10, inf
from cv2 import CAP_PROP_FRAME_WIDTH, CAP_PROP_FRAME_HEIGHT, CAP_PROP_FPS
from multiprocessing.pool import ThreadPool
from collections import deque
from utils.utilities import separate_channels
from utils.common_metrics import calc_psnr, calc_ws_psnr, calc_ssim, calc_nrmse

num_frames = 0
avg_value = 0


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

    return (raw_width == coded_width) and (raw_height == coded_height)


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

    _num_threads = int(cv.getNumberOfCPUs() / 2)
    _pool = ThreadPool(processes=_num_threads)
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

    while True:

        # process generated tasks
        while len(_pending) > 0 and _pending[0].ready():
            value, frame_time = _pending.popleft().get()
            _latency.update(_clock() - frame_time)
            print("latency        :  %.1f ms" % (_latency.value * 1000))
            print("frame interval :  %.1f ms" % (_frame_interval.value * 1000))
            print("PSNR Value     :  %.3f [dB]" % value)
            print("Frame count    :  {0}\n".format(_frame_count))
            _avg_value += value
            _frame_count += 1

        # if length of dequeue is smaller than number of available threads -> start generating new tasks
        if len(_pending) < _num_threads:
            # read frames
            has_raw_frames, raw_frame = _cap_raw.read()
            has_coded_frames, coded_frame = _cap_coded.read()

            # update frame interval and latency
            t = _clock()
            _frame_interval.update(t - _last_frame_time)
            _last_frame_time = t

            if _colorSpaceType == "YUV":

                # _raw_channels, _coded_channels = separate_channels(raw_frame, coded_frame, _colorSpaceType)
                # generate new asynchronous task
                _yuv_raw = cv.cvtColor(raw_frame, cv.COLOR_BGR2YCrCb)
                _yuv_coded = cv.cvtColor(coded_frame, cv.COLOR_BGR2YCrCb)

                task = _pool.apply_async(calc_psnr, (_yuv_raw, _yuv_coded, t))
            else:
                task = _pool.apply_async(calc_psnr, (raw_frame, coded_frame, t))

            # append task to queue
            _pending.append(task)

        if cv.waitKey(1) == ord('q'):
            break

    print('calculation finished')

    print("average {0} value    :  {1}".format(_metricToCalculate, _avg_value))
