import numpy as np
import numpy.linalg as npl
import cv2
import sys
from PIL import Image
from numpy.lib.stride_tricks import sliding_window_view
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

        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #Shi-Tomasi
        corners = cv2.goodFeaturesToTrack(img, 10000, 0.001, 10)

        for p in np.int0(corners):
            x, y = p.ravel()
            print(x, y)
            cv2.circle(img, (x,y), 3, 255, -1 )
        corners = corners[:, 0, :]
        corners[:, 0], corners[:, 1] = corners[:, 1].copy(), corners[:, 0].copy()
        
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

        R_candidates = [R0, R1, R2, R3]
        for i, sol in enumerate(R_candidates):
            det = npl.det(sol)
            if not det > 0:
                break
            print('--------check R', i)
            print(f"det(R): {det}")
            print(f"R: {sol}\n")

        print(f"E:\n{E}\ntX@R:{tX @ R1}\n")
        return R1, t

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

        #######
        # tensor_depth = h*w
        # warp_tensor = np.repeat(Warp[:, :, np.newaxis], tensor_depth, axis=2)
        # print('---------')
        # # warp tensor shape: (3, 3, 515616)
        # # warp tensor shape: (3, 3, 476532)

        # pixel_xs = np.array([i for i in range(w)])
        # pixel_ys = np.array([j for j in range(h)])

        # img_tensor = np.ones((3, 1, tensor_depth))
        # for i in range(w): #1312
        #     img_tensor[0, 0, w:w+h] = np.ones((1, 1, h))*i
        #     img_tensor[1, 0, w:w+h] = pixel_ys.copy()


        # print(f"warp tensor result: {img_tensor.shape}")
        
        # result = warp_tensor @ img_tensor
        
            

        ######
        for i in range(w): #1312
            for j in range(h): #393
                p = Warp @ np.array([i, j, 1])
                p = p.astype(int)
                # check within boundary
                if (0<= p[0] < w) and (0<= p[1] < h):
                    rect_img[j, i, :] = img[p[1], p[0], :]
        return rect_img


    def rectify_image(self, R, Rrect, K1, K2, left_img, right_img):
        R1 = Rrect
        R2 = R @ Rrect

        left_rect_img = self.backward_warping(R1, K1, left_img)
        right_rect_img = self.backward_warping(R2, K2, right_img)
        return left_rect_img, right_rect_img
    

    def get_disparity_map(self, F, left_img, right_img):
        h, w = left_img.shape[:2]
        disparity_map = np.zeros((h, w, 3), dtype=np.uint8)

        epi_line = np.zeros(3).astype(int)
        
        print('-------------disparity')
        f_half = 3 #filter half size
        stride = 1
        pad = 0

        #TODO: backward-warping style.... from right to left. 
        f_size = 5 
        tensor_depth = right_img.shape[1]
        target_tensor = np.zeros((f_size, f_size, 3, tensor_depth))

        tensor_4d = np.zeros((f_size, f_size, 3, tensor_depth))
        tensor_5d = np.zeros((f_size, f_size, 3, tensor_depth, tensor_depth))

        filters_tensor_4d = tensor_4d.copy()            
        target_tensor_4d = tensor_4d.copy()

        filters_tensor_5d = tensor_5d.copy()            
        target_tensor_5d = tensor_5d.copy()

        disparity_map = np.zeros((h, w))
        pixel_indexes = np.array(range(w))
        for start_h in range(0, h - f_size - 1):

            from_img = left_img[start_h:start_h + f_size, :, :]
            to_img = right_img[start_h:start_h + f_size, :, :]

            for i in range(tensor_depth - f_size):
                filters_tensor_4d[:, :, :, i] = from_img[:, i : i + f_size, :]
                target_tensor_4d[:, :, :, i] = to_img[:, i : i + f_size, :]

            filters_tensor_5d = np.repeat(filters_tensor_4d[:, :, :,:, np.newaxis], tensor_depth, axis=4)
            target_tensor_5d = np.repeat(target_tensor_4d[:, :, :,:, np.newaxis], tensor_depth, axis=4)

            filters_tensor_5d = np.transpose(filters_tensor_5d, [0, 1, 2, 4, 3])
            result = target_tensor_5d - filters_tensor_5d
            result = np.abs(np.sum(result, axis=(0,1,2)))
            result = np.argmin(result, axis=0)

            disparity_map[start_h, :] = pixel_indexes - result

        return disparity_map


