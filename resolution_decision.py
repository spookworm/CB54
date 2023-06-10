from IPython import get_ipython
import matplotlib.pyplot as plt
from skimage import data
import numpy as np

# Clear workspace
get_ipython().run_line_magic('reset', '-sf')

image = data.horse()
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
plt.imshow(np.flipud(image), origin='lower')
plt.show()

"""
According to Wolfram Alpha:
    height of a domestic horse: 1.8 m
    length of a domestic horse: 2.4 m
"""
dim_y = 1.8  # print(image.shape[0]) # height
dim_x = 2.4  # print(image.shape[1]) # length


def scale_geometry(image, dim_x, dim_y):
    scale_geometry_x = dim_x / image.shape[1]
    scale_geometry_y = dim_y / image.shape[0]
    return min(scale_geometry_x, scale_geometry_y)


scale_geometry = scale_geometry(image, dim_x, dim_y)

print(scale_geometry)

