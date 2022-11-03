import numpy as np
import numpy.linalg as npl
import cv2
import sys
from PIL import Image
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


    def detect_feature_points(self, img):
        if self.opencv:
            # Detector parameters
            block_size = 2
            aperture_size = 5
            k = 0.2
            epsilon = 0.03

            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = np.float32(img)
            dst = cv2.cornerHarris(img, block_size, aperture_size, k)

            return np.array(np.where(dst > epsilon*dst.max())).T #shape: (n, 2)



    def get_correspondence_points(self, left_points, right_points):
        left_size = left_points.shape[0]
        right_size = right_points.shape[0]

        if left_size < right_size:
            P = left_points #reference
            Q = right_points
            ref_size = left_size
            isLeftRef = True
        else: #left_size >= right_size
            P = right_points #reference
            Q = left_points
            ref_size = right_size
            isLeftRef = False

        left_correspondents = np.zeros((ref_size, 2), dtype=np.int16)
        right_correspondents = np.zeros((ref_size, 2), dtype=np.int16)
        for i in range(ref_size):
                ref_point = P[i, :] #(x,y)
                distances = npl.norm(Q - ref_point, axis=1)
                dist_min_point = Q[np.argmin(distances), :]
                if isLeftRef:
                    left_correspondents[i, :] = np.array([ref_point], dtype=np.int16)
                    right_correspondents[i, :] = np.array([dist_min_point], dtype=np.int16)
                else: #reference == right
                    left_correspondents[i, :] = np.array([dist_min_point], dtype=np.int16)
                    right_correspondents[i, :] = np.array([ref_point], dtype=np.int16)
        
        return left_correspondents, right_correspondents
        
    def get_F_matrix(self, left_points, right_points):
        F, _ = cv2.findFundamentalMat(left_points, right_points, cv2.FM_LMEDS)
        return F

    def get_E_matrix(self, F, K1, K2):
        return K2.T @ F @ K1

    def decomp_E_matrix(self, E):
        U, Sigma, VT = npl.svd(E, full_matrices = True) 
        Sigma = np.diag(Sigma)
        V = VT.T

        W = np.array([[0, -1, 0],
                    [1, 0, 0], 
                    [0, 0, 1]])

        R = U @ W.T @ V.T #W.T = npl.inv(W)

        t_tilde_x = U @ W @ Sigma @ U.T
        t_tilde = np.array([[t_tilde_x[2][0], t_tilde_x[0][1], t_tilde_x[1][0]]]).T
        t = - R.T @ t_tilde # R.T = npl.inv(R)

        return R, t

    def estimate_Rrect(self, t):
        e1 = t / npl.norm(t)
        e1 = e1.T[0] # shape: (3, )
        e2 = np.array([-t[1][0], t[0][0], 0]) / npl.norm(t[:2])
        e3 = np.cross(e1, e2)

        print('e1,e2,e3 norms:', npl.norm(e1), npl.norm(e2), npl.norm(e3))
        print('e1,e2,e3 shapes:', e1.shape, e2.shape, e3.shape)
        print('e1,e2,e3 :', e1, e2, e3)
        Rrect = np.array([e1, e2, e3])
        print('det(Rrect): ', npl.det(Rrect))
        return Rrect
        
    def backward_warping(self, R, K, img):
        h, w = img.shape[:2]
        rect_img = np.zeros((h, w, 3), dtype=np.uint8)
        Warp = K @ npl.inv(R) @ npl.inv(K)
        p = np.zeros(3).astype(int)
        for i in range(h): #393
            for j in range(w): #1312
                p = Warp @ np.array([j, i, 1])
                p = p.astype(int)
                try: 
                    # print(f'i:{i} j:{j} p[0]:{p[0]} p[1]:{p[1]}')
                    rect_img[i, j, :] = img[p[1], p[0], :]
                except IndexError as e:
                    pass
                    # print(e)
        return rect_img


    def rectify_image(self, R, Rrect, K1, K2, left_img, right_img):
        #Theory
        # R1 = Rrect
        # R2 = R @ Rrect

        R1 = R
        R2 = Rrect

        left_rect_img = self.backward_warping(R1, K1, left_img)
        right_rect_img = self.backward_warping(R2, K2, right_img)

        # left_rect_img = scipy.misc.toimage(left_rect_img)
        # right_rect_img = scipy.misc.toimage(right_rect_img)
        # print(f"mapping: \n{left_map.shape}")

        #         pixel = left_img[i, j]
        #         x = np.array([i, j , 1])
        #         x_prime = R1 @ x
        #         x_prime = x_prime / x_prime[2]

        #         print('xprimeshape ', x_prime)
                # rotated[i, j] = 

        return left_rect_img, right_rect_img