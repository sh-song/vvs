import numpy as np

mat = np.zeros((11,22,33,44,55))
print(mat.shape)
mat = np.transpose(mat,[0, 1, 2, 4, 3] )
print(mat.shape)
