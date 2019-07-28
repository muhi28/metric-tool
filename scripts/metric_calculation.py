import cv2
import argparse
import sys
import os
import matplotlib.pyplot as plt
from scripts.metric_calculator import MetricCalculator

"""
    check resolution of of video to compare
"""
def check_video_resolutions(raw_cap, coded_cap):

    raw_width = int(raw_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    raw_height = int(raw_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    coded_width = int(coded_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    coded_height = int(coded_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    return (raw_width == coded_width) and (raw_height == coded_height)


if __name__ == '__main__':

    if len(sys.argv) <= 1:
        print("Update the values -> raw video file/image and coded video file / image")
        exit(0)

    # extract parameters
    rawFilePath = sys.argv[1]  # get raw file path
    codedFilePath = sys.argv[2]  # get coded file path
    colorSpaceType = sys.argv[3]  # get selected color space to calculate the metrics
    metricToCalculate = sys.argv[4]  # get selected metric to calculate

    # initialize video capture for raw and coded video
    video_cap_raw = cv2.VideoCapture(rawFilePath)
    video_cap_coded = cv2.VideoCapture(codedFilePath)

    # parse basename
    _, raw_file_basename = os.path.split(rawFilePath)
    _, coded_file_basename = os.path.split(codedFilePath)

    print("Selected color space -> {0}".format(colorSpaceType))
    print("Selected raw video file -> {0}".format(raw_file_basename))
    print("Selected coded video file -> {0}".format(coded_file_basename))
    print("FPS -> {0}".format(video_cap_coded.get(cv2.CAP_PROP_FPS)))
    print("Color Space -> {0}".format(colorSpaceType))
    print("Selected Metric -> {0}".format(metricToCalculate))

    # check if video streams are opened
    if not video_cap_raw.isOpened():
        print("Could not open raw video file")
        exit(-1)

    if not video_cap_coded.isOpened():
        print("Could not open coded video file!!!")
        exit(-1)

    # check whether the selected video files have the same resolution or not
    if not check_video_resolutions(video_cap_raw, video_cap_coded):
        print("Video resolutions doesn't match")

    metric_calc = MetricCalculator(video_cap_raw, video_cap_coded, colorSpaceType)

    frames, metric_data = metric_calc.perform_measuring(selected_metric=metricToCalculate)

    print("AVG-{0}-VALUE -> {1:.3f} [dB]".format(metricToCalculate, metric_calc.get_avg_value()))

    plt.plot(frames, metric_data)
    plt.xlabel("Frame Number")
    plt.ylabel("{0}-Value [dB]".format(metricToCalculate))
    plt.show()
