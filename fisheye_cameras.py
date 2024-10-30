import numpy as np
import cv2


class ELP:
    def __init__(self, fov_x=80, fov_y=50):
        self.fov_x = fov_x
        self.fov_y = fov_y

    def undistort_fisheye(self, img):
        frame_height, frame_width = img.shape[:2]
        DIM = (frame_width, frame_height)
        K = np.array(
            [
                [479.33387664063054, 0.0, 329.2747865040309],
                [0.0, 478.9978764351638, 240.3858301417113],
                [0.0, 0.0, 1.0],
            ]
        )
        D = np.array(
            [
                [-0.09606337518419901],
                [0.007195909715586325],
                [-0.11058122583794522],
                [0.1267928190814884],
            ]
        )
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(
            K, D, np.eye(3), K, DIM, cv2.CV_16SC2
        )
        undistorted_img = cv2.remap(
            img,
            map1,
            map2,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        )
        return undistorted_img
