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
    target_data = np.array(img1, dtype=np.float64)
    ref_data = np.array(img2, dtype=np.float64)

    diff = ref_data - target_data
    diff = diff.flatten('C')

    mse = math.sqrt(np.mean(diff ** 2.))

    return 20 * math.log10(255 / mse)


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

    frameNumber = 0
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

    psnrVal = 0

    # start frame extraction and metric calculation
    while True:

        # read frames
        raw_ret, raw_frame = video_cap_raw.read()
        cod_ret, coded_frame = video_cap_coded.read()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # check if the return value for frame capturing is false -> this means that we have reached the end of the video
        if not raw_ret or not cod_ret or frameNumber == 300:
            break

        # check selected color space
        if colorSpaceType == "RGB":

            psnrVal = psnr(raw_frame, coded_frame)

            print("FRAME {0}".format(frameNumber))
            print("PSNR-VALUE [RGB] -> {0:.2f} [dB] ".format(psnrVal))

        elif colorSpaceType == "YUV":

            # calculate Y-PSNR
            yuvRaw = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2YCrCb)
            yuvCoded = cv2.cvtColor(coded_frame, cv2.COLOR_BGR2YCrCb)

            # extract Y [Luminance] channel
            y_raw, _, _ = cv2.split(yuvRaw)
            y_coded, _, _ = cv2.split(yuvCoded)

            # cv2.imshow("Y-Raw", y_raw)
            # cv2.imshow("Y-Coded", y_coded)

            # perform psnr for y-channel
            psnrVal = psnr(yuvRaw, yuvCoded)
            # upsnr = psnr(u_raw, u_coded)
            # vpsnr = psnr(v_raw, v_coded)

            # print current value
            print("FRAME {0}".format(frameNumber))
            print("PSNR-VALUE [Y-Channel] -> {0:.2f} [dB] ".format(psnrVal))


        else:
            print("Wrong color space.....")
            exit(-1)

        # if there are black frames inside our test sequences then the psnr for them would be infinity
        # therefore we need to check if there are some infinity values to skip them
        if psnrVal == math.inf:
            continue

        # append data to frames and psnr list
        frames.append(frameNumber)
        psnrValues.append(psnrVal)

        frameNumber += 1
        avg_psnr += psnrVal

    # release used video capture and destroy all opened windows
    video_cap_raw.release()
    video_cap_coded.release()
    cv2.destroyAllWindows()

    print("AVG-PSNR-VALUE -> {0:.3f} [dB]".format(avg_psnr / frameNumber))

    plt.plot(frames, psnrValues)
    plt.xlabel("Frame Number")
    plt.ylabel("{0}-Value [dB]".format(metricToCalculate))
    plt.show()
