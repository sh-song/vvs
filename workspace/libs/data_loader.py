import os
import cv2
class DataLoader:
    def __init__(self, cfg):
        self.cfg = cfg

    def sort_filenames(self, filenames):
        #Init sorted list
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
        # Get filenames
        filenames = os.listdir(path)

        # Sort filenames
        filenames = self.sort_filenames(filenames)

        # Init image list
        imgs = [None] * len(filenames)
         
        # Read images into image list
        for i, filename in enumerate(filenames):
            imgs[i] = cv2.imread(f"{path}/{filename}")
            print(f"{path}/{filename}")

        # Check images properly loaded
        if imgs[-1] is not None:
            print(f"[Loader] Loaded images in {path}")
        else:
            raise Exception(f"[Loader] Failed to load images in {path}")

        return imgs
