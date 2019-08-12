import argparse
import sys
import os
import time
import matplotlib.pyplot as plt
from scripts.metric_calculator import MetricCalculator
from cv2 import VideoCapture, CAP_PROP_FRAME_WIDTH, CAP_PROP_FRAME_HEIGHT, CAP_PROP_FPS


def __init_argparser():
    # construct the argument parser
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
    args = __init_argparser()

    # extract parameters
    rawFilePath = args["raw"]  # get raw file path
    codedFilePath = args["encoded"]  # get coded file path
    colorSpaceType = args["colorspace"]  # get selected color space to calculate the metrics
    metricToCalculate = args["metric"]  # get selected metric to calculate

    # initialize video capture for raw and coded video
    video_cap_raw = VideoCapture(rawFilePath)
    video_cap_coded = VideoCapture(codedFilePath)

    # init metric calculator
    metric_calc = MetricCalculator(video_cap_raw, video_cap_coded, colorSpaceType)

    # parse basename
    _, raw_file_basename = os.path.split(rawFilePath)
    _, coded_file_basename = os.path.split(codedFilePath)

    print("Selected color space -> {0}".format(colorSpaceType))
    print("Selected raw video file -> {0}".format(raw_file_basename))
    print("Selected coded video file -> {0}".format(coded_file_basename))
    print("FPS -> {0}".format(video_cap_coded.get(CAP_PROP_FPS)))
    print("Color Space -> {0}".format(colorSpaceType))
    print("Selected Metric -> {0}\n".format(metricToCalculate))

    # check if video streams are opened
    if not video_cap_raw.isOpened() or not video_cap_coded.isOpened():
        print("Could not open raw video file")
        exit(-1)

    # check whether the selected video files have the same resolution or not
    if not __check_video_resolutions(video_cap_raw, video_cap_coded):
        print("Video resolutions doesn't match")

    start = time.time()

    frames, metric_data = metric_calc.perform_measuring(selected_metric=metricToCalculate)

    duration = time.time() - start

    print("Duration of Measuring -> {0} sec.".format(duration))
    print("AVG-{0}-VALUE -> {1:.3f} [dB]".format(metricToCalculate, metric_calc.get_avg_value()))

    plt.plot(frames, metric_data)
    plt.xlabel("Frame Number")
    plt.ylabel("{0}-Value [dB]".format(metricToCalculate))
    plt.show()
