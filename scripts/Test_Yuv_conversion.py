import cv2
import numpy as np
import math
import sys

def psnr(img1, img2):
    target_data = np.array(img1, dtype=np.float64)
    ref_data = np.array(img2, dtype=np.float64)

    diff = ref_data - target_data
    diff = diff.flatten('C')

    mse = math.sqrt(np.mean(diff ** 2.))

    return 20 * math.log10(255 / mse)

def make_lut_u():
    return np.array([[[i,255-i,0] for i in range(256)]],dtype=np.uint8)

def make_lut_v():
    return np.array([[[0,255-i,i] for i in range(256)]],dtype=np.uint8)


if __name__ == '__main__':

    if len(sys.argv) <= 1:
        print("Update the values -> raw video file/image and coded video file / image")
        exit(0)

    rawFilePath = sys.argv[1]  # get raw file path
    codedFilePath = sys.argv[2]  # get coded file path

    # initialize video capture for raw and coded video
    video_cap_raw = cv2.VideoCapture(rawFilePath)
    video_cap_coded = cv2.VideoCapture(codedFilePath)

    frameNumber = 0
    avg_psnr = 0

    frames = []  # create frame list used for our plotting part
    psnrValues = []  # used to store our frame by frame psnr values

    while True:

        # read frames
        raw_ret, raw_frame = video_cap_raw.read()
        cod_ret, coded_frame = video_cap_coded.read()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # check if the return value for frame capturing is false -> this means that we have reached the end of the video
        if not raw_ret or not cod_ret or frameNumber == 300:
            break

            # calculate Y-PSNR
        yuvRaw = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2YCrCb)
        yuvCoded = cv2.cvtColor(coded_frame, cv2.COLOR_BGR2YCrCb)

        # extract Y [Luminance] channel
        y_raw, ur, vr = cv2.split(yuvRaw)
        y_coded, uc, vc = cv2.split(yuvCoded)


            # cv2.imshow("Y-Raw", y_raw)
            # cv2.imshow("Y-Coded", y_coded)

        ut_u, lut_v = make_lut_u(), make_lut_v()

        y = cv2.cvtColor(y_raw, cv2.COLOR_GRAY2BGR)
        u = cv2.cvtColor(ur, cv2.COLOR_GRAY2BGR)
        v = cv2.cvtColor(vr, cv2.COLOR_GRAY2BGR)

            # perform psnr for y-channel
        psnrVal = psnr(yuvRaw, yuvCoded)
            # upsnr = psnr(u_raw, u_coded)
            # vpsnr = psnr(v_raw, v_coded)

            # print current value
        print("FRAME {0}".format(frameNumber))
        print("PSNR-VALUE [Y-Channel] -> {0:.2f} [dB] ".format(psnrVal))


        # if there are black frames inside our test sequences then the psnr for them would be infinity
        # therefore we need to check if there are some infinity values to skip them
        if psnrVal == math.inf:
            continue

        # append data to frames and psnr list
        frames.append(frameNumber)
        psnrValues.append(psnrVal)

        frameNumber += 1

    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    y, u, v = cv2.split(img_yuv)

    lut_u, lut_v = make_lut_u(), make_lut_v()

    # Convert back to BGR so we can apply the LUT and stack the images
    y = cv2.cvtColor(y, cv2.COLOR_GRAY2BGR)
    u = cv2.cvtColor(u, cv2.COLOR_GRAY2BGR)
    v = cv2.cvtColor(v, cv2.COLOR_GRAY2BGR)

    u_mapped = cv2.LUT(u, lut_u)
    v_mapped = cv2.LUT(v, lut_v)
