import numpy as np
class Config:
    K = np.array([[9.842439e+02, 0.000000e+00, 6.900000e+02],
                [0.000000e+00, 9.808141e+02, 2.331966e+02],
                [0.000000e+00, 0.000000e+00, 1.000000e+00]])
    distortion = np.array([-3.728755e-01, 2.037299e-01, 2.219027e-03, 1.383707e-03,-7.233722e-02])

    left_images = 'data/target_2011_09_26_drive_0048/unsync_unrect/image_02/data'
    right_images = 'data/target_2011_09_26_drive_0048/unsync_unrect/image_03/data'
    save_path = 'output'