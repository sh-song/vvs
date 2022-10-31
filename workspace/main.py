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
        left_imgs = self.loader.left_images()
        right_imgs = self.loader.right_images()

        # Undistort images
        left_imgs = self.vvs.undistort_images(left_imgs)
        right_imgs = self.vvs.undistort_images(right_imgs)

        # Load calibration data
        left_calib_imgs = self.loader.left_calib_images()
        right_calib_imgs = self.loader.right_calib_images()

        # Feature Point Detection
        left_feature_points = self.vvs.detect_feature_points(left_calib_imgs)
        right_feature_points = self.vvs.detect_feature_points(right_calib_imgs)

        # Get Correspondence
        correspondents = self.vvs.get_correspondence_points(left_feature_points, right_feature_points)

        # Get Fundamental Matrix
        F = self.vvs.get_F_matrix(correspondents)

        # Get Essential Matrix
        E = self.vvs.get_E_matrix(F, K)

        # Get R, t between left/right cameras
        R, t = self.vvs.decomp_E_matrix(E) 

        # Estimate Rrect
        Rrect = self.vvs.estimate_Rrect(t)

        # Rectify image
        left_imgs, right_imgs = self.vvs.rectify_image(R, Rrect, left_imgs, right_imgs)

        # Get disparity map (matching)
        disparity_map = self.vvs.get_disparity_map(left_imgs, right_imgs)
        
        # Save result        
        # self.saver.save_images(self.right_imgs, 'right_undistorted')
    
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