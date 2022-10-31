import numpy as np
import cv2
class VVS:
    def __init__(self, cfg):
        self.cfg = cfg
        self.opencv = False

    def allow_opencv(self, flag:bool):
        self.opencv = flag

    def undistort_images(self, K, dist, imgs):
        K = K
        dist = dist

        undistorted_imgs = [None] * len(imgs)
        if self.opencv:
            h,  w = imgs[0].shape[:2]
            new_K, roi = cv2.getOptimalNewCameraMatrix(K, dist, (w,h), 1, (w,h))
            x, y, w, h = roi

            for i, img in enumerate(imgs):
                undistorted_imgs[i] = cv2.undistort(img, K, dist, None, new_K)[y:y+h, x:x+w]

        print(f"[VVS] Undistorted images")
        return undistorted_imgs