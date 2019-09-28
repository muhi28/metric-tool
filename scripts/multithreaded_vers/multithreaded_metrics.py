import argparse
import gc
import os
import sys
from time import time

import cv2 as cv
import numpy as np

from utils.parse_video_files import get_video_files
from utils.common_metrics import calc_psnr, calc_ssim, calc_wpsnr, calc_ws_psnr
from cv2 import split, CAP_PROP_FRAME_COUNT
from multiprocessing.pool import Pool, ThreadPool
from collections import deque
from math import log10, cos, pi, inf

# variables defining maximal pixel value and progressbar length
MAX_PIXEL = 255
bar_len = 60


def progress(count, total, status=''):

    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    sys.stdout.write('[%s] %s%s ...%s\r' % (bar, percents, '%', status))
    sys.stdout.flush()


def __init_argparser():
    """
        used to generate and initialize the argument parser
    :return: dict containing the available arguments
    """
    # create arg parser
    arg_parser = argparse.ArgumentParser()

    # add argument elements to argparse
    arg_parser.add_argument("-r", "--raw", required=True, help="original input video")
    arg_parser.add_argument("-e", "--encoded", required=True, help="encoded input videos")
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
        "PSNR": calc_psnr,
        "WS-PSNR": calc_ws_psnr,
        "SSIM": calc_ssim,
        "W-PSNR": calc_wpsnr
    }

    # get the selected metric to calculate
    m = switcher.get(selected_metric, calc_psnr)

    return m


def print_progress(iteration, total):
    """
        shows a progress bar during video processing
    :param iteration:
    :param total:
    :return:
    """
    filled_len = int(round(bar_len * iteration / float(total)))

    percents = round(102.3 * iteration / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    sys.stdout.write('\r[%s] %s%s ...%s' % (bar, percents, '%', "Processing Progress"))
    sys.stdout.flush()


def perform_processing(num_processes, raw_file_path, coded_files_path, metric) -> None:
    """
        perform the metric calculation

    :param num_processes: number of available processes
    :param raw_file_path: raw video file path
    :param coded_files_path: coded video file path
    :param metric: selected metric

    """
    # define metric which would be calculated
    _metric_func = __get_metric(_metricToCalculate)

    # high performance object used to cache async tasks
    _task_buffer = deque()

    # start calculation timer
    start_time = time()

    # open a pool of processes used to calculate the selected metric
    with Pool(processes=_num_processes) as _pool:

        # parse encoded video files from given directory
        encoded_files = get_video_files(coded_files_path)

        # perform calculation for each given encoded file
        for encoded_file in encoded_files:

            # init necessary stuff

            _frame_count = 1
            _avg_value = 0.0

            # set current raw video capture
            _cap_raw = cv.VideoCapture(raw_file_path)

            # set current encoded video capture
            _cap_coded = cv.VideoCapture(encoded_file)

            # get number of frames to process
            num_frames = _cap_raw.get(cv.CAP_PROP_FRAME_COUNT)

            # cut out the video name from the given video path
            _, _coded_file_basename = os.path.split(encoded_file)
            print("Selected coded video file -> {0}\n".format(_coded_file_basename))

            # start the calculation
            while True:

                # process generated tasks
                while len(_task_buffer) > 0 and _task_buffer[0].ready():
                    # print_progress(_frame_count, num_frames)

                    # pop element from rightmost side
                    value = _task_buffer.pop().get()

                    # print current calculation
                    print("PSNR Value     :  %.3f [dB]" % value)

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
                        break

                    # check whether the raw and coded videos are of same shape
                    # otherwise continue to next encoded file
                    elif raw_frame.shape != coded_frame.shape:
                        print("video shape doesn't match...")
                        break

                    # check whether YUV or RGB, etc. color space is selected
                    if _colorSpaceType == "YUV":

                        # _raw_channels, _coded_channels = separate_channels(raw_frame, coded_frame, _colorSpaceType)
                        # generate new asynchronous task
                        _yuv_raw = cv.cvtColor(raw_frame, cv.COLOR_BGR2YCrCb)
                        _yuv_coded = cv.cvtColor(coded_frame, cv.COLOR_BGR2YCrCb)

                        # check which metric is selected
                        if metric in {"PSNR", "WS-PSNR"}:
                            task = _pool.apply_async(_metric_func, (split(_yuv_raw)[0], split(_yuv_coded)[0]))
                        else:
                            task = _pool.apply_async(_metric_func, (_yuv_raw, _yuv_coded))

                    else:
                        # if selected color space is RGB, etc. -> then calculate the metric using all 3 channels
                        # combined
                        task = _pool.apply_async(_metric_func, (raw_frame, coded_frame))

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
    _num_processes = int(cv.getNumberOfCPUs() / 2)

    print("Start calculation ....\n")

    print("Settings:")
    print("Number of processes    :  {0}".format(_num_processes))
    print("Color Space -> {0}".format(_colorSpaceType))
    print("Selected Metric -> {0}".format(_metricToCalculate))
    print("Selected raw video file -> {0}".format(_raw_file_basename))

    # start the video processing part
    perform_processing(_num_processes, _rawFilePath, _codedFilesPath, _metricToCalculate)
