import numpy as np
import cv2
class VVS:
    def __init__(self, cfg):
        self.cfg = cfg
        self.opencv = False

    def allow_opencv(self, flag:bool):
        self.opencv = flag

    def undistort_images(self, K, dist, imgs):
        # K = K
        # dist = dist

        undistorted_imgs = [None] * len(imgs)
        if self.opencv:
            h,  w = imgs[0].shape[:2]
            new_K, roi = cv2.getOptimalNewCameraMatrix(K, dist, (w,h), 1, (w,h))
            x, y, w, h = roi

            for i, img in enumerate(imgs):
                undistorted_imgs[i] = cv2.undistort(img, K, dist, None, new_K)[y:y+h, x:x+w]

        print(f"[VVS] Undistorted images")
        return undistorted_imgs


    def detect_feature_points(self, imgs):
        feature_points = [None] * len(imgs)
        if self.opencv:
            # Detector parameters
            block_size = 2
            aperture_size = 5
            k = 0.2
            epsilon = 0.03

            for i, img in enumerate(imgs):
                img2 = img.copy()
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img = np.float32(img)
                dst = cv2.cornerHarris(img, block_size, aperture_size, k)
                #dst = cv2.dilate(dst, None)
                print(img.shape)
                img2[dst > epsilon*dst.max()] = [0,0,255]
                feature_points[i] = img2

            return feature_points