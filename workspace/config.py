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

    #left_images = 'data/target_2011_09_26_drive_0048/unsync_unrect/image_02/data'
    #right_images = 'data/target_2011_09_26_drive_0048/unsync_unrect/image_03/data'
    left_images = 'data/devel/target_2011_09_26_drive_0048/unsync_unrect/image_02/data'
    right_images = 'data/devel/target_2011_09_26_drive_0048/unsync_unrect/image_03/data'


    left_calib_images = 'data/calib_2011_09_26_drive_0119/image_02/data'
    right_calib_images = 'data/calib_2011_09_26_drive_0119/image_03/data'
    save_path = 'output'


    R_left_color = np.array([[9.999758e-01, -5.267463e-03, -4.552439e-03], 
                            [5.251945e-03, 9.999804e-01, -3.413835e-03], 
                            [4.570332e-03, 3.389843e-03, 9.999838e-01]])

    R_right_color = np.array([[9.995599e-01, 1.699522e-02, -2.431313e-02], 
                            [-1.704422e-02, 9.998531e-01, -1.809756e-03], 
                            [2.427880e-02, 2.223358e-03, 9.997028e-01]])

    R_rect_left_color = np.array([[9.998817e-01, 1.511453e-02, -2.841595e-03], 
                                [-1.511724e-02, 9.998853e-01, -9.338510e-04], 
                                [2.827154e-03, 9.766976e-04, 9.999955e-01]])

    R_rect_right_color = np.array([[9.998321e-01, -7.193136e-03, 1.685599e-02], 
                                [7.232804e-03, 9.999712e-01, -2.293585e-03], 
                                [-1.683901e-02, 2.415116e-03, 9.998553e-01]])


