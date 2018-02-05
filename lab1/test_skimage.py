import numpy as np
import matplotlib.pyplot as plt

from skimage import data

# camera = data.camera()
# print(type(camera))
# print(camera.shape)
#
# # inds_r = np.arange(len(camera))
# # print(inds_r)
# # # from 1 to 512
# # inds_c = 4 * inds_r % len(camera)
# # print(inds_c)
# # camera[inds_r, inds_c] = 0
#
#
# plt.imshow(camera)
# plt.show()

arr = np.array([])
arr = np.hstack((arr, np.array([1,2,3])))
# arr is now [1,2,3]

arr = np.vstack((arr, np.array([4,5,6])))
# arr is now [[1,2,3],[4,5,6]]
print(arr)
