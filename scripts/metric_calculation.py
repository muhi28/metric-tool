import cv2
import numpy as np
import math
import sys
import os
import matplotlib.pyplot as plt


def make_lut_u():
    return np.array([[[i, 255-i, 0] for i in range(256)]], dtype=np.uint8)


def make_lut_v():
    return np.array([[[0, 255-i, i] for i in range(256)]], dtype=np.uint8)


def psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return math.inf

    pixel_max = 255.0
    return 10 * math.log10((pixel_max * pixel_max) / mse)


def checkVideoResolutions(raw_cap, coded_cap):

    raw_width = int(raw_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    raw_height = int(raw_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    coded_width = int(coded_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    coded_height = int(coded_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    return (raw_width == coded_width) and (raw_height == coded_height)


if __name__ == '__main__':

    if len(sys.argv) <= 1:
        print("Update the values -> raw video file/image and coded video file / image")
        exit(0)

    rawFilePath = sys.argv[1]  # get raw file path
    codedFilePath = sys.argv[2]  # get coded file path
    colorSpaceType = sys.argv[3]  # get selected color space to calculate the metrics
    metricToCalculate = sys.argv[4] # get selected metric to calculate

    video_cap_raw = cv2.VideoCapture(rawFilePath)
    video_cap_coded = cv2.VideoCapture(codedFilePath)


    # video_cap_raw = cv2.VideoCapture("E:\\HEVC_HM_Software\\software\\hm_360\\Tiling_Workflow_5g\\tiling_test_workflow\\test_hevc_tiles\\videos\\explore_the_world\\explore.mp4")
    # video_cap_coded = cv2.VideoCapture("E:\\HEVC_HM_Software\\software\\hm_360\\Tiling_Workflow_5g\\tiling_test_workflow\\test_hevc_tiles\\videos\\explore_the_world\\explore_encoded.mp4")
    _, raw_file_basename = os.path.split(rawFilePath)
    _, coded_file_basename = os.path.split(codedFilePath)

    print("Selected color space -> {0}".format(colorSpaceType))
    print("Selected raw video file -> {0}".format(raw_file_basename))
    print("Selected coded video file -> {0}".format(coded_file_basename))
    print("FPS -> {0}".format(video_cap_coded.get(cv2.CAP_PROP_FPS)))
    print("Selected Metric -> {0}".format(metricToCalculate))

    count = 0
    avg_psnr = 0

    frames = []  # create frame list used for our plotting part
    psnrValues = []  # used to store our frame by frame psnr values

    # check if video streams are opened
    if not video_cap_raw.isOpened():
        print("Could not open raw video file")
        exit(-1)

    if not video_cap_coded.isOpened():
        print("Could not open coded video file!!!")
        exit(-1)

    # check whether the selected video files have the same resolution or not
    if not checkVideoResolutions(video_cap_raw, video_cap_coded):
        print("Video resolutions doesn't match")

    while True:

        raw_ret, raw_frame = video_cap_raw.read()
        cod_ret, coded_frame = video_cap_coded.read()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # check if the return value for frame capturing is false -> this means that we have reached the end of the video
        if not raw_ret or not cod_ret or count == 300:
            break

        psnrVal = psnr(raw_frame, coded_frame)

        # if there are black frames inside our test sequences then the psnr for them would be infinity
        # therefore we need to check if there are some infinity values to skip them
        if psnrVal == math.inf:
            continue

        frames.append(count)
        psnrValues.append(psnrVal)

        # print("FRAME {0}".format(count))
        # print("PSNR-VALUE [RGB] -> {0:.2f} [dB] ".format(psnrVal))
        # count += 1
        # avg_psnr += psnrVal

        lut_u = make_lut_u()
        lut_v = make_lut_v()

        test = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2YUV)

        raw_y, raw_u, raw_v = cv2.split(test)

        raw_y = cv2.cvtColor(raw_y, cv2.COLOR_GRAY2BGR)
        raw_u = cv2.cvtColor(raw_u, cv2.COLOR_GRAY2BGR)
        raw_v = cv2.cvtColor(raw_v, cv2.COLOR_GRAY2BGR)

        u = cv2.LUT(raw_u, lut_u)
        v = cv2.LUT(raw_v, lut_v)

        cv2.imshow("Y-Value", raw_y)
        cv2.imshow("U-Value", u)
        cv2.imshow("V-Value", v)

    video_cap_raw.release()
    video_cap_coded.release()
    cv2.destroyAllWindows()

    print("AVG-PSNR-VALUE -> {0:.3f} [dB]".format(avg_psnr / count))

    plt.plot(frames, psnrValues)
    plt.xlabel("Frame Number")
    plt.ylabel("{0}-Value [dB]".format(metricToCalculate))
    plt.show()
