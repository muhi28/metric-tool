import numpy as np
import math
import cv2


class MetricCalculator:

    def __init__(self, video_cap_raw, video_cap_coded, colorspacetype):
        self.video_cap_raw = video_cap_raw
        self.video_cap_coded = video_cap_coded
        self.colorspacetype = colorspacetype
        self.frames = []
        self.dataCollection = []
        self.avgValue = 0

    def __calcpsnr(self, img1, img2):

        target_data = np.array(img1, dtype=np.float64)
        ref_data = np.array(img2, dtype=np.float64)

        diff = ref_data - target_data
        diff = diff.flatten('C')

        mse = math.sqrt(np.mean(diff ** 2.))

        return 20 * math.log10(255 / mse)

    def performVideoExtraction(self):

        psnrVal = 0
        framenumber = 1

        while True:

            raw_ret, raw_frame = self.video_cap_raw.read()
            cod_ret, coded_frame = self.video_cap_coded.read()

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

                # check if the return value for frame capturing is false -> this means that we have reached the end of the video
            if not raw_ret or not cod_ret or framenumber == 15:
                break

                # check selected color space
            if self.colorspacetype == "RGB":

                psnrVal = self.calcpsnr(raw_frame, coded_frame)

                print("FRAME {0}".format(framenumber))
                print("PSNR-VALUE [RGB] -> {0:.2f} [dB] ".format(psnrVal))

            elif self.colorspacetype == "YUV":

                # calculate Y-PSNR
                yuvRaw = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2YCrCb)
                yuvCoded = cv2.cvtColor(coded_frame, cv2.COLOR_BGR2YCrCb)

                # extract Y [Luminance] channel
                y_raw, _, _ = cv2.split(yuvRaw)
                y_coded, _, _ = cv2.split(yuvCoded)

                # cv2.imshow("Y-Raw", y_raw)
                # cv2.imshow("Y-Coded", y_coded)

                # perform psnr for y-channel
                psnrVal = self.__calcpsnr(yuvRaw, yuvCoded)
                # upsnr = psnr(u_raw, u_coded)
                # vpsnr = psnr(v_raw, v_coded)

                # print current value
                print("FRAME {0}".format(framenumber))
                print("PSNR-VALUE [Y-Channel] -> {0:.2f} [dB] ".format(psnrVal))


            else:
                print("Wrong color space.....")
                exit(-1)

                # if there are black frames inside our test sequences then the psnr for them would be infinity
                # therefore we need to check if there are some infinity values to skip them
            if psnrVal == math.inf:
                continue

                # append data to frames and psnr list
            self.frames.append(framenumber)
            self.dataCollection.append(psnrVal)

            framenumber += 1
            self.avgValue += psnrVal

        self.video_cap_raw.release()
        self.video_cap_coded.release()

        return self.frames, self.dataCollection

    def getAvgValue(self):
        return self.avgValue / len(self.frames)
