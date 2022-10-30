#!/usr/bin/python3
import argparse
from scv import SCV as scv
import cv2
import os
from config import Config
from libs.dataloader import DataLoader
from libs.output_saver import OutputSaver
class StereoMatching:
    def __init__(self, cfg):
        self.cfg = cfg

        #is opencv allowed
        self.isAllowed = False

        self.loader = DataLoader(cfg)
        self.saver = OutputSaver(cfg)

        self.left_imgs = None
        self.right_imgs = None
        
    def allow_opencv(self, flag: bool):
        self.isAllowed = flag
        print(f"[Main] Allow OpenCV: {flag}")

    def run(self):
        self.left_imgs = self.loader.left_images()
        self.right_imgs = self.loader.right_images()

        if self.isAllowed:
        #Undistort TODO: make undistort_image module
            K = self.cfg.K
            dist = self.cfg.distortion
            h,  w = self.left_imgs[0].shape[:2]
            new_K, roi = cv2.getOptimalNewCameraMatrix(K, dist, (w,h), 1, (w,h))
            x, y, w, h = roi
            for i, img in enumerate(self.left_imgs):
                self.left_imgs[i] = cv2.undistort(img, K, dist, None, new_K)[y:y+h, x:x+w]
            for i, img in enumerate(self.right_imgs):
                self.right_imgs[i] = cv2.undistort(img, K, dist, None, new_K)[y:y+h, x:x+w]


            self.saver.save_images(self.right_imgs, 'right_undistorted')
    
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