#!/usr/bin/python3
import argparse
import cv2
import os
from config import Config
from libs.data_loader import DataLoader
from libs.output_saver import OutputSaver
from libs.midterm import VVS

class StereoMatching:
    def __init__(self, cfg):
        self.cfg = cfg

        self.loader = DataLoader(cfg)
        self.saver = OutputSaver(cfg)
        self.vvs = VVS(cfg)

    def allow_opencv(self, flag: bool):
        self.vvs.allow_opencv(flag)
        print(f"[Main] Allow OpenCV: {flag}")

    def run(self):
        # Load data
        left_imgs = self.loader.load_images(self.cfg.left_images)
        right_imgs = self.loader.load_images(self.cfg.right_images)

        left_calib_imgs = self.loader.load_images(self.cfg.left_calib_images)
        right_calib_imgs = self.loader.load_images(self.cfg.right_calib_images)

        # Undistort images
        K1 = self.cfg.K_left_color
        K2 = self.cfg.K_right_color
        dist1 = self.cfg.dist_left_color
        dist2 = self.cfg.dist_right_color

        left_imgs = self.vvs.undistort_images(K1, dist1, left_imgs)
        right_imgs = self.vvs.undistort_images(K2, dist2, right_imgs)

        left_calib_imgs = self.vvs.undistort_images(K1, dist1, left_calib_imgs)
        right_calib_imgs = self.vvs.undistort_images(K2, dist2, right_calib_imgs)



       # Feature Point Detection

        left_feature_points = self.vvs.detect_feature_points(left_calib_imgs)
        right_feature_points = self.vvs.detect_feature_points(right_calib_imgs)


        # Get Correspondence
        # correspondents = self.vvs.get_correspondence_points(left_feature_points, right_feature_points)

        # Get Fundamental Matrix
        # F = self.vvs.get_F_matrix(correspondents)

        # Get Essential Matrix
        # E = self.vvs.get_E_matrix(F, self.cfg.K)

        # Get R, t between left/right cameras
        # R, t = self.vvs.decomp_E_matrix(E) 

        # Estimate Rrect
        # Rrect = self.vvs.estimate_Rrect(t)

        # Rectify image
        R = None
        Rrect = None
        left_imgs, right_imgs = self.vvs.rectify_image(R, Rrect, left_imgs, right_imgs)
        self.saver.save_images(left_imgs, 'left_rectified')
        self.saver.save_images(right_imgs, 'right_rectified')
        exit(0)

        # Get disparity map (matching)
        disparity_map = self.vvs.get_disparity_map(left_imgs, right_imgs)
        
        # Save result        
    
if __name__ == "__main__":

    #Set python argument parser
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--opencv', default='n')
    args = argparser.parse_args()

    #Initialize
    cfg = Config()
    stereo = StereoMatching(cfg)

    #Check if OpenCV allowed
    flag = True if (args.opencv == 'y') else False
    stereo.allow_opencv(flag)

    #Run
    stereo.run()