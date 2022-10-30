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

        #is opencv allowed
        self.isAllowed = False

        self.loader = DataLoader(cfg)
        self.saver = OutputSaver(cfg)
        self.vvs = VVS(cfg)

        self.left_imgs = None
        self.right_imgs = None
        
    def allow_opencv(self, flag: bool):
        self.vvs.allow_opencv(flag)
        print(f"[Main] Allow OpenCV: {flag}")

    def run(self):
        self.left_imgs = self.loader.left_images()
        self.right_imgs = self.loader.right_images()

        #Undistort images
        self.left_imgs = self.vvs.undistort_images(\
                        self.cfg.K, self.cfg.dist, self.left_imgs)
        self.right_imgs = self.vvs.undistort_images(\
                        self.cfg.K, self.cfg.dist, self.right_imgs)

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