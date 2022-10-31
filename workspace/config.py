import numpy as np
class Config:
    K_left_color = np.array([[9.597910e+02, 0.000000e+00, 6.960217e+02], 
                            [0.000000e+00, 9.569251e+02, 2.241806e+02], 
                            [0.000000e+00, 0.000000e+00, 1.000000e+00]])
    dist_left_color = np.array([-3.691481e-01, 1.968681e-01, 1.353473e-03, 5.677587e-04, -6.770705e-02])

    K_right_color = np.array([[9.037596e+02, 0.000000e+00, 6.957519e+02], 
                            [0.000000e+00, 9.019653e+02, 2.242509e+02], 
                            [0.000000e+00, 0.000000e+00, 1.000000e+00]])
    dist_right_color = np.array([-3.639558e-01, 1.788651e-01, 6.029694e-04, -3.922424e-04, -5.382460e-02])

    left_images = 'data/target_2011_09_26_drive_0048/unsync_unrect/image_02/data'
    right_images = 'data/target_2011_09_26_drive_0048/unsync_unrect/image_03/data'

    left_calib_images = 'data/calib_2011_09_26_drive_0119/image_02/data'
    right_calib_images = 'data/calib_2011_09_26_drive_0119/image_03/data'
    save_path = 'output'
