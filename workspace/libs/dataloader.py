import os
import cv2
class DataLoader:
    def __init__(self, cfg):
        self.cfg = cfg

    def load_images(self, path):
        filenames = os.listdir(path)
        path += '/'
        imgs = [None] * len(filenames)
         
        for i, filename in enumerate(filenames):
            imgs[i] = cv2.imread(path + filename)
        return imgs

    def left_images(self):
        path = self.cfg.left_images
        imgs = self.load_images(path)
        if imgs[-1] is not None:
            print(f"[Loader] Loaded left images in {path}")
        else:
            raise Exception(f"[Loader] Failed to load left images in {path}")
        return imgs

    def right_images(self):
        path = self.cfg.right_images
        imgs = self.load_images(path)
        if imgs[-1] is not None:
            print(f"[Loader] Loaded right images in {path}")
        else:
            raise Exception(f"[Loader] Failed to load right images in {path}")
        return imgs       