import math
from math import log10, inf, cos, pi
from cv2.cv2 import split
import skimage.measure
import numpy as np

from utils.vector_util import Vector3, get_yaw, get_pitch

MAX_PIXEL = 255


# _viewport = Viewport()
#
#
# def set_viewport_settings(vp_width, vp_height, vp_fov_x):
#     global _viewport
#
#     _viewport.set_width(vp_width)
#     _viewport.set_height(vp_height)
#     _viewport.set_fov_x(vp_fov_x)


def calc_ssim(img1, img2, multi_channel=False):
    """
        calculates the structural similarity between two images
    :param multi_channel: checks if we are operating on more than one color channel
    :rtype: float
    :param img1: raw image (original)
    :param img2: coded image
    :return: ssim value

    The following settings are necessary to match the implementation of Wang et. al. [1]

    References:
    --------

    .. [1] Wang, Z., Bovik, A. C., Sheikh, H. R., & Simoncelli, E. P.
       (2004). Image quality assessment: From error visibility to
       structural similarity. IEEE Transactions on Image Processing,
       13, 600-612.
       https://ece.uwaterloo.ca/~z70wang/publications/ssim.pdf,
       :DOI:`10.1109/TIP.2003.819861`
    """

    return skimage.measure.compare_ssim(img1, img2, gaussian_weights=True,
                                        sigma=1.5, use_sample_covariance=False, multichannel=multi_channel)


def calc_psnr(img1, img2):
    """
        calculates the peak-signal-to noise ration between two images
    :param img1: original image
    :param img2: coded image
    :return: psnr value
    """

    height, width = img1.shape[0], img1.shape[1]

    target_data = np.array(img2, dtype=np.float64)
    ref_data = np.array(img1, dtype=np.float64)

    diff = ref_data - target_data
    diff = diff.flatten('C')

    mse = np.sum(diff ** 2) / (height * width)

    # if black frames appear during the measurement (leading to mse=zero), return the maximum float value for them.
    if mse == 0:
        return inf

    return 10 * log10((MAX_PIXEL ** 2) / mse)


def calc_wpsnr(img1, img2):
    """
        calculate weighted psnr value
    :param img1: raw image
    :param img2: coded image
    :return: w-psnr value
    """
    val1_raw, val2_raw, val3_raw = split(img1)
    val1_coded, val2_coded, val3_coded = split(img2)

    y_psnr = calc_psnr(val1_raw, val1_coded)
    u_psnr = calc_psnr(val2_raw, val2_coded)
    v_psnr = calc_psnr(val3_raw, val3_coded)

    return ((6 * y_psnr) + u_psnr + v_psnr) / 8.0


def calc_vpsnr(img1, img2, viewing_direction, _viewport):

    height, width = img1.shape[0], img1.shape[1]

    y_raw, y_coded = img1, img2

    sum = 0.0

    vp_neg, vp_pos = (-1 * int(_viewport.get_width() / 2)), int(_viewport.get_width() / 2)

    print(f"viewport range: {vp_neg} to {vp_pos}")

    for i in range(vp_neg, vp_pos):

        for j in range(vp_neg, vp_pos):
            res = _viewport.get_spherical_coords(Vector3(i, j, 0), viewing_direction)

            yaw = get_yaw(res)
            pitch = get_pitch(res)

            x_factor = width / 360.0
            y_factor = height / 180.0

            x = int(round(yaw * x_factor) % int(width))
            y = int(round((pitch + 90) * y_factor) % int(height))

            pixel_raw = int(y_raw[x][y])
            pixel_coded = int(y_coded[x][y])

            y_channel = int(pixel_raw - pixel_coded)

            sum += y_channel * y_channel

    _vpsnr_mse = (sum / (width * height))

    if _vpsnr_mse == 0:
        return inf

    return 10.0 * np.log10((MAX_PIXEL * MAX_PIXEL) / _vpsnr_mse)







def calc_nrmse(img1, img2, norm_type="min-max"):
    """
        calculate normalized root mean-squared error (NRMSE)

    :param img1: raw image
    :param img2: coded image
    :param norm_type: selected normalization type
    :return: nrmse value
    """
    return skimage.measure.compare_nrmse(img1, img2, norm_type)


def calc_ws_psnr(img1, img2):
    """
        calculate weighted spherical psnr
    :param img1: raw image
    :param img2: coded image
    :return: ws-psnr value
    """

    frame_height, frame_width = img1.shape[0], img1.shape[1]

    _ref_vals = np.array(img1, dtype=np.float64)
    _target_vals = np.array(img2, dtype=np.float64)

    _sum_val = _w_sum = 0.0

    _diff = _ref_vals - _target_vals
    _diff = np.abs(_diff) ** 2
    _diff = _diff.flatten('C')

    _pixel_weights = [cos((j - (frame_height / 2) + 0.5) * (pi / frame_height))
                      for j in range(frame_height)]

    counter = 0
    weight_id = 0

    for val in _diff:

        _sum_val += val * _pixel_weights[weight_id]
        _w_sum += _pixel_weights[weight_id]

        if counter == frame_width:
            counter = 0
            weight_id += 1

        counter += 1

    _sum_val = _sum_val / _w_sum

    if _sum_val == 0:
        _sum_val = 100
    else:
        _sum_val = 10 * log10((MAX_PIXEL * MAX_PIXEL) / _sum_val)

    return _sum_val
