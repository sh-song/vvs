import os
import cv2
class DataLoader:
    def __init__(self, cfg):
        self.cfg = cfg

    def sort_filenames(self, filenames):
        #Init sorted lsit
        sorted = [(None, None)] * len(filenames) 

        #Extract frame number from filename
        for i, filename in enumerate(filenames):
            frame = filename.split('.')[0]
            sorted[i] = (int(frame), filename)

        #Sort in ascending order
        sorted.sort()

        # Return filenames in sorted order
        for i in range(len(filenames)):
            filenames[i] = sorted[i][1]

        return filenames

    def load_images(self, path):
        filenames = os.listdir(path)
        filenames = self.sort_filenames(filenames)
        imgs = [None] * len(filenames)
         
        for i, filename in enumerate(filenames):
            imgs[i] = cv2.imread(f"{path}/{filename}")
            print(f"{path}/{filename}")
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