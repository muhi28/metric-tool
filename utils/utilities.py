from cv2 import split, cvtColor, COLOR_BGR2HSV, COLOR_BGR2YCrCb


def separate_channels(raw_frame, coded_frame, color_space_type):
    """
        used to convert color space from RGB to YUV if needed and to separate the channels
    :param raw_frame: original image
    :param coded_frame: coded image
    :param color_space_type: selected color space
    :return: separated channels for selected color space
    """

    raw_channels = []
    coded_channels = []

    if color_space_type == "RGB":
        # extract r,g,b channels and calculate metric for each channel
        raw_channels = split(raw_frame)
        coded_channels = split(coded_frame)

    elif color_space_type == "YUV":
        raw_yuv = cvtColor(raw_frame, COLOR_BGR2YCrCb)
        coded_yuv = cvtColor(coded_frame, COLOR_BGR2YCrCb)

        # extract Y [Luminance] channel
        raw_channels = split(raw_yuv)
        coded_channels = split(coded_yuv)

    elif color_space_type == "HVS":
        raw_hsv = cvtColor(raw_frame, COLOR_BGR2HSV)
        coded_hsv = cvtColor(coded_frame, COLOR_BGR2HSV)

        raw_channels = split(raw_hsv)
        coded_channels = split(coded_hsv)

    else:
        print("Wrong color space selected!!!")
        exit(-1)

    return raw_channels, coded_channels
