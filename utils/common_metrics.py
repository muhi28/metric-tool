from math import log10, inf, cos, pi
from skimage.measure import compare_ssim, compare_nrmse

import numpy as np

frame_width = 0
frame_height = 0
MAX_PIXEL = 255


def init_frame_data(height, width):
    global frame_width
    global frame_height

    frame_height = height
    frame_width = width


def calc_ssim(img1, img2, multi_channel):
    """
        calculates the structural similarity between two images
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

    return compare_ssim(img1, img2, gaussian_weights=True,
                        sigma=1.5, use_sample_covariance=False, multichannel=multi_channel)


def calc_psnr(img1, img2):
    """
        calculates the peak-signal-to noise ration between two images
    :param img1: original image
    :param img2: coded image
    :return: psnr value
    """

    target_data = np.array(img2, dtype=np.float64)
    ref_data = np.array(img1, dtype=np.float64)

    diff = ref_data - target_data
    diff = diff.flatten('C')

    mse = np.sum(diff ** 2) / (frame_width * frame_height)

    # if black frames appear during the measurement (leading to mse=zero), return the maximum float value for them.
    if mse == 0:
        return inf

    return 10 * log10((MAX_PIXEL ** 2) / mse)


def calc_nrmse(img1, img2, norm_type):
    """
        calculate normalized root mean-squared error (NRMSE)

    :param img1: raw image
    :param img2: coded image
    :return: nrmse value
    """
    return compare_nrmse(img1, img2, norm_type)


def calc_ws_psnr(img1, img2):
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
