import numpy as np
import cv2
import sys
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
                # dst = cv2.dilate(dst, None)

                # feature_points[i] = np.array(np.where(dst > epsilon*dst.max())).T
                feature_points[i] = np.array(np.where(dst > epsilon*dst.max()))

            return feature_points


    def get_correspondence_points(self, left_feature_points, right_feature_points):
        length = len(left_feature_points)
        def get_correspondence_indices(P, Q):
            """For each point in P find closest one in Q."""
            p_size = P.shape[1]
            q_size = Q.shape[1]
            correspondences = []
            for i in range(p_size):
                p_point = P[:, i]
                min_dist = sys.maxsize
                chosen_idx = -1
                for j in range(q_size):
                    q_point = Q[:, j]
                    dist = np.linalg.norm(q_point - p_point)
                    if dist < min_dist:
                        min_dist = dist
                        chosen_idx = j
                correspondences.append((i, chosen_idx))
            return correspondences
        for i in range(length):
            left = left_feature_points[i]
            right = right_feature_points[i]

            corres = get_correspondence_indices(left, right)
        return corres
    def rectify_image(self, R, Rrect, left_imgs, right_imgs):
        #TODO: calculate R1, R2 from R, Rrect
        R1 = self.cfg.R_rect_left_color
        R2 = self.cfg.R_rect_right_color
        #######
        h1, w1 = left_imgs[0].shape[:2]
        h2, w2 = right_imgs[0].shape[:2]

        length = len(left_imgs)
        rect_left_imgs = [None] * length
        rect_right_imgs = [None] * length
        center = (w1/2, h1/2)
        rotate_matrix = cv2.getRotationMatrix2D(center=center, angle=45, scale=1)
        print(rotate_matrix.shape)
        print('R1', R1.shape)
 
        for i in range(length):
            rect_left_imgs[i] = cv2.warpPerspective(src=left_imgs[i], M=R1, dsize=(w1, h1))
            rect_right_imgs[i] = cv2.warpPerspective(src=right_imgs[i],M=R2,dsize=(w2, h2))
            # rect_left_imgs[i] = R1@left_imgs[i]
            # rect_right_imgs[i] = R2@right_imgs[i]

        return rect_left_imgs, rect_right_imgs