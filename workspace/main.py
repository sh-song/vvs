#!/usr/bin/python3
import argparse
from scv import SCV as scv
import cv2

class StereoMatching:
    def __init__(self):
        #is opencv allowed
        self.isAllowed = False
        self.save_path = ''
        self.left_img = None
        self.right_img = None

    def allow_opencv(self, flag: bool):
        self.isAllowed = flag

    def load_left_image(self, path):
        if self.isAllowed:
            self.left_image = cv2.imread(path)

    def load_right_image(self, path):
        if self.isAllowed:
            self.right_image = cv2.imread(path)

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
        '--left_image',
        default='',
        help='image path'
    )

    argparser.add_argument(
        '--right_image',
        default='',
        help='image path'
    )
    argparser.add_argument(
        '--result_save',
        default='',
        help='result save path'
    )

    args = argparser.parse_args()

    #Load
    stereo = StereoMatching()
    if args.opencv == "y":
        stereo.allow_opencv(True)
        print('Using OpenCV')

    elif args.opencv == "n":
        stereo.allow_opencv(False)
        print('Using sh-song implementation')
    
    else:
        print("[Error] Argument parsing failed")

    stereo.load_left_image(args.left_image)
    stereo.load_right_image(args.right_image)
    stereo.set_result_save_path(args.result_save)

    #Run
    stereo.run()