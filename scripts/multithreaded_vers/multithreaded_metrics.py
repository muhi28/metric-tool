import os
import cv2 as cv

from scripts.multithreaded_vers.metric_calculator import MetricCalculator
from tkinter import Tk  # from tkinter import Tk for Python 3.x
from tkinter.filedialog import askopenfilename, askdirectory
from tkinter.simpledialog import askstring, askinteger

if __name__ == '__main__':
    # select the original image
    Tk().withdraw()  # we don't want a full GUI, so keep the root window from appearing
    _rawFilePath = askopenfilename(title="Select original video file")  # show an "Open" dialog box and return the path
    # to the selected file
    _codedFilesPath = askdirectory(title="Select encoded video file folder")
    _colorSpaceType = askstring(title="Set color space", prompt="Color Space:")
    _metricToCalculate = askstring(title="Select metric type", prompt="Metric:")
    _num_frames_skip = int(askinteger(title="Select number of frames to skip at the beginning"
                                      , prompt="Number of frames:"))

    # parse raw basename
    _, _raw_file_basename = os.path.split(_rawFilePath)

    # set number of processes
    _num_processes = int(cv.getNumberOfCPUs())

    print("Start calculation ....\n")

    print("Settings:")
    print(f"Number of processes    :  {_num_processes}")
    print(f"Color Space -> {_colorSpaceType}")
    print(f"Selected Metric -> {_metricToCalculate}")
    print(f"Selected raw video file -> {_raw_file_basename}")

    # start the video processing part
    # perform_processing(_num_processes, _num_frames_skip, _rawFilePath, _codedFilesPath, _metricToCalculate)
    metric_proc = MetricCalculator(_num_processes, _num_frames_skip,
                                   _rawFilePath, _codedFilesPath,
                                   _colorSpaceType, _metricToCalculate)

    metric_proc.start_processing()
