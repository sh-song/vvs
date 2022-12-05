#!/usr/bin/python3
from config import Config
from libs.data_loader import DataLoader
from libs.output_saver import OutputSaver
from libs.vvs import VVS
import numpy as np
class StereoMatching:
    def __init__(self, cfg):
        self.cfg = cfg

        self.loader = DataLoader(cfg)
        self.saver = OutputSaver(cfg)
        self.vvs = VVS(cfg)

    def run(self):
        # Load data
        left_imgs = self.loader.load_images(self.cfg.left_images)
        right_imgs = self.loader.load_images(self.cfg.right_images)

        left_calib_imgs = self.loader.load_images(self.cfg.left_calib_images)
        right_calib_imgs = self.loader.load_images(self.cfg.right_calib_images)

        
        # Get parameters
        K1 = self.cfg.K_left_color
        K2 = self.cfg.K_right_color
        dist1 = self.cfg.dist_left_color
        dist2 = self.cfg.dist_right_color

        len_imgs = len(left_imgs)
        len_calib_imgs = len(left_calib_imgs)

        # Undistort images

        left_undistorted_imgs = [None] * len_imgs 
        right_undistorted_imgs = [None] * len_imgs 

        for i in range(len_imgs):
            left_undistorted_imgs[i] = self.vvs.undistort_image(K1, dist1, left_imgs[i])
            right_undistorted_imgs[i] = self.vvs.undistort_image(K2, dist2, right_imgs[i])


        left_undistorted_calib_imgs = [None] * len_calib_imgs 
        right_undistorted_calib_imgs = [None] * len_calib_imgs 

        for i in range(len_calib_imgs):
            left_undistorted_calib_imgs[i] = self.vvs.undistort_image(K1, dist1, left_calib_imgs[i])
            right_undistorted_calib_imgs[i] = self.vvs.undistort_image(K2, dist2, right_calib_imgs[i])


        # Feature Point Detection
        left_feature_points_list = [None] * len_calib_imgs 
        right_feature_points_list = [None] * len_calib_imgs 

        left_feature_img_list = [None] * len_calib_imgs 
        right_feature_img_list = [None] * len_calib_imgs 

        for i in range(len_calib_imgs):
            left_feature_points_list[i], left_feature_img_list[i] = self.vvs.detect_feature_points(left_undistorted_calib_imgs[i])
            right_feature_points_list[i], right_feature_img_list[i] = self.vvs.detect_feature_points(right_undistorted_calib_imgs[i])

        print(f"Get feature points...\n")


        # Get Correspondence
        left_correspondence_points_list = [None] * len_calib_imgs
        right_correspondence_points_list = [None] * len_calib_imgs
        
        for i in range(len_calib_imgs):
            left_correspondence_points_list[i], right_correspondence_points_list[i] = \
                self.vvs.get_correspondence_points(left_feature_points_list[i], \
                                                    right_feature_points_list[i])

        left_correspondence_points = np.concatenate(left_correspondence_points_list)
        right_correspondence_points = np.concatenate(right_correspondence_points_list)

        print(f"Get correspondence points...\n")


        # Get Fundamental Matrix
        F = self.vvs.get_F_matrix(left_correspondence_points, right_correspondence_points)
        print("F: \n", F)


        # Get Essential Matrix
        E = self.vvs.get_E_matrix(F, self.cfg.K_left_color, self.cfg.K_right_color)
        print("E: \n", E)


        # Get R, t between left/right cameras
        R, t = self.vvs.decomp_E_matrix(E) 
        print("R: \n", R)
        print("t: \n", t)


        # Estimate Rrect
        Rrect = self.vvs.estimate_Rrect(t)
        print("Rrect: \n", Rrect)


        # Rectify image
        left_rect_imgs = [None] * len_imgs
        right_rect_imgs = [None] * len_imgs

        for i in range(len_imgs):
            left_rect_imgs[i], right_rect_imgs[i] = \
                self.vvs.rectify_image(R, Rrect, K1, K2, left_undistorted_imgs[i], right_undistorted_imgs[i])
       
        # Get disparity map (matching)
        disparity_maps = [None] * len_imgs
        for i in range(len_imgs):
            disparity_maps[i] = self.vvs.get_disparity_map(F, left_imgs[i], right_imgs[i])

        # Save result        
        self.saver.save_images(disparity_maps, 'disparity_maps')
    
if __name__ == "__main__":

    #Initialize
    cfg = Config()
    stereo = StereoMatching(cfg)

    #Run
    stereo.run()