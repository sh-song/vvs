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

            k = 0.05

            epsilon = 0.005

            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            #Shi-Tomasi
            corners = cv2.goodFeaturesToTrack(img, 10000, 0.001, 10)
            print('----------------')
            print(np.int0(corners[:, 0, :]))
            
            # img = np.float32(img)
            # dst = cv2.cornerHarris(img, block_size, aperture_size, k)
            # harris = np.array(np.where(dst > epsilon*dst.max())).T #shape: (n, 2)
            # print('-sibal')
            # print(harris)

            print('----------------')
             
            for p in np.int0(corners):
                x, y = p.ravel()
                print(x, y)
                cv2.circle(img, (x,y), 3, 255, -1 )
            corners = corners[:, 0, :]
            # corners[:, 0], corners[:, 1] = corners[:, 1].copy(), corners[:, 0].copy()
            print(corners)
           
            return np.int0(corners), img


    def get_correspondence_points(self, left_points, right_points):
        left_size = left_points.shape[0]
        right_size = right_points.shape[0]

        if left_size <= right_size:
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
        # left_points[:, 0], left_points[:, 1] = left_points[:, 1].copy(), left_points[:, 0].copy()
        # right_points[:, 0], right_points[:, 1] = right_points[:, 1].copy(), right_points[:, 0].copy()
        F, _ = cv2.findFundamentalMat(left_points, right_points, cv2.USAC_ACCURATE)
        return F

    
    def get_E_matrix(self, F, K1, K2):
        
        return K2.T @ F @ K1

    def get_cross_product_operator(self, n):
        n_x = n[0]
        n_y = n[1]
        n_z = n[2]

        return np.array([[0, -n_z, n_y],
                        [n_z, 0, -n_x],
                        [-n_y, n_x, 0]])

    def decomp_cross_product_operator(self, nX):
        n_x = nX[2, 1]
        n_y = nX[0, 2]
        n_z = nX[1, 0]
        return np.array([n_x, n_y, n_z])
        
    def decomp_E_matrix(self, E):
        U, Sigma, VT = npl.svd(E, full_matrices = True) 
        Sigma = np.diag(Sigma)
        t = U[:, 2]
        V = VT.T
        print('--------------')
        print(f"U: \n{U}\nSigma:\n{Sigma}\nVT:\n{VT}\n")
        print(f"t==U[:,2]:\n{t}\n")
        print('--------------')

        tX = self.get_cross_product_operator(t)

        #W == Rz_(+90)
        #W.T == Rz_(-90)
        W = np.array([[0, -1, 0],
                    [1, 0, 0], 
                    [0, 0, 1]], dtype=np.float64)
        
        R0 = + U @ W.T @ V.T
        R1 = + U @ W @ V.T
        R2 = - U @ W.T @ V.T
        R3 = - U @ W @ V.T

        # tX1 = U @ W @ Sigma @ U.T
        # tX2 = U @ W.T @ Sigma @ U.T
        # print(f"tX:\n{tX}\ntX1:\n{tX1}\ntX2:\n{tX2}")
        # R1 = U @ W.T @ V.T #W.T = npl.inv(W)
        # R2 = U @ W @ V.T #W.T = npl.inv(W)

        # t_tilde_x_1 = U @ W @ Sigma @ U.T
        # t_tilde_1 = np.array([[t_tilde_x_1[2][1], t_tilde_x_1[0][2], t_tilde_x_1[1][0]]]).T

        # t1_1 = - R1.T @ t_tilde_1 # R.T = npl.inv(R)
        # t1_2 = - R2.T @ t_tilde_1 # R.T = npl.inv(R)

        # t_tilde_x_2 = U @ W.T @ Sigma @ U.T
        # t_tilde_2 = np.array([[t_tilde_x_2[2][1], t_tilde_x_2[0][2], t_tilde_x_2[1][0]]]).T
        # print(f"tilde1: {t_tilde_x_1}\ntilde2: {t_tilde_x_2}")
        # t2_1 = - R1.T @ t_tilde_2 # R.T = npl.inv(R)
        # t2_2 = - R2.T @ t_tilde_2 # R.T = npl.inv(R)
        # solutions = [(R1, t1_1), (R1, t2_1), (R2, t1_2), (R2, t2_2)]
        R_candidates = [R0, R1, R2, R3]
        for i, sol in enumerate(R_candidates):
            det = npl.det(sol)
            if not det > 0:
                break
            print('--------check R', i)
            print(f"det(R): {det}")
            print(f"R: {sol}\n")

        print(f"E:\n{E}\ntX@R:{tX @ R1}\n")
        return R0, t

    def estimate_Rrect(self, t): #TODO: CHECK CHECK CHECK
        e1 = t / npl.norm(t)
        e2 = np.array([-t[1], t[0], 0]) / npl.norm(t[:2])
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
                    rect_img[i, j, :] = img[p[1], p[0], :]
                except IndexError as e:
                    pass
                    # print(e)
        return rect_img


    def rectify_image(self, R, Rrect, K1, K2, left_img, right_img):
        #Theory
        R1 = Rrect
        R2 = R @ Rrect

        # R1 = R
        # R2 = R
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