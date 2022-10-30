#!/usr/bin/python3
import argparse
from scv import SCV as scv
import cv2
import os
from config import Config
class StereoMatching:
    def __init__(self, config):
        #is opencv allowed
        self.isAllowed = False
        self.save_path = ''
        self.left_imgs = None
        self.right_imgs = None
        

    def allow_opencv(self, flag: bool):
        self.isAllowed = flag

    def load_images(self, path):
        filenames = os.listdir(path)
        path += '/'
        imgs = [None] * len(filenames)
         
        for i, filename in enumerate(filenames):
            if self.isAllowed:
                imgs[i] = cv2.imread(path + filename)

        return imgs
                
    #TODO Seperate Loader in dataloader.py
    def load_left_images(self, path):
        self.left_images = self.load_images(path)
        print(f"[Loader] Loaded left images in {path}")
    def load_right_images(self, path):
        self.right_images = self.load_images(path)
        print(f"[Loader] Loaded right images in {path}")

    def set_result_save_path(self, path):
        self.save_path = path

    def run(self):
        
        pass
    
if __name__ == "__main__":

    #Argument parse
    argparser = argparse.ArgumentParser(
        description="12200595 sh-song "
    )

    argparser.add_argument(
        '--opencv',
        default='n',
        help='y or n'
    )

    argparser.add_argument(
        '--left_images',
        default='',
        help='image path'
    )

    argparser.add_argument(
        '--right_images',
        default='',
        help='image path'
    )
    argparser.add_argument(
        '--save_path',
        default='',
        help='result save path'
    )

    args = argparser.parse_args()

    #Load
    cfg = Config()
    stereo = StereoMatching(cfg)
    if args.opencv == "y":
        stereo.allow_opencv(True)
        print('Using OpenCV')

    elif args.opencv == "n":
        stereo.allow_opencv(False)
        print('Using sh-song implementation')
    
    else:
        print("[Error] Argument parsing failed")

    #TODO remove argparser and do the same on cfg
    stereo.load_left_images(args.left_images)
    stereo.load_right_images(args.right_images)
    stereo.set_result_save_path(args.save_path)

    #Run
    stereo.run()