import cv2
from unittest.mock import Mock, patch
from unittest import TestCase

import utils.common_metrics as uc


class TestCommonModule(TestCase):

    def setUp(self) -> None:
        self.raw_path = "/home/muhamed/Documents/WORK_ITEC/ITEC-Project/videos/video_sequences/shower_falls" \
                        "/crystal_falls_120frames.mp4 "
        self.coded_path = "/home/muhamed/Documents/WORK_ITEC/ITEC-Project/videos/video_sequences/shower_falls" \
                          "/Crystal_Shower_Falls-GOP-8-IP-8-Tiling-6x4/str0-converted.mp4"

        self.image1 = cv2.VideoCapture(self.raw_path).read()
        self.image2 = cv2.VideoCapture(self.coded_path).read()

    @patch("skimage.measure")
    def test_calc_ssim(self, mock_skimage):
        uc.calc_ssim(self.image1, self.image2, True)
        mock_skimage.compare_ssim.assert_called_once()

    @patch("skimage.measure")
    def test_calc_nrmse(self, mock_nrmse):
        uc.calc_nrmse(self.image1, self.image2, "min-max")
        mock_nrmse.compare_nrmse.assert_called_once()
