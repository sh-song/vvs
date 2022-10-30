#!/usr/bin/python3
import argparse
from scv import SCV as scv
import cv2
import os
from config import Config
from libs.dataloader import DataLoader
class StereoMatching:
    def __init__(self, cfg):
        self.cfg = cfg

        #is opencv allowed
        self.isAllowed = False

        self.loader = DataLoader(cfg)

        self.left_imgs = None
        self.right_imgs = None
        
    def allow_opencv(self, flag: bool):
        self.isAllowed = flag
        print(f"[Main] Allow OpenCV: {flag}")

    def run(self):
        self.left_imgs = self.loader.left_images()
        self.right_imgs = self.loader.right_images()
    
if __name__ == "__main__":

    #Argument parse
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--opencv', default='n')
    args = argparser.parse_args()

    #Load
    cfg = Config()
    stereo = StereoMatching(cfg)

    flag = True if (args.opencv == 'y') else False
    stereo.allow_opencv(flag)

    #Run
    stereo.run()