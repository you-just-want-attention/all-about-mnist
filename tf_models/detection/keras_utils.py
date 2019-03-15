import numpy as np

"""
Two Style Cordinates Representation

1. minmax style :
[batch_size, x_min, x_max, y_min, y_max]

x_min : the leftmost coordinates of an object
x_max : the rightmost coordinates of an object
y_min : the uppermost coordinates of an object
y_max : the lowest coordinates of an object

2. center style : 
[batch_size, center_x, center_y, width, height]

center_x : center x coordinates of an object
center_y : center y coordinates of an object
width  : the log scale about the width of object 
height : the log scale about the height of object


minmax2center : minmax style -> center style
center2minmax : center style -> minmax style
"""


def minmax2center(points):
    if points.ndim == 2:
        center_x = points[:, :2].sum(axis=1) / 2
        center_y = points[:, 2:].sum(axis=1) / 2

        width = points[:, 1] - points[:, 0]
        height = points[:, 3] - points[:, 2]
        return np.stack([center_x, center_y, width, height], axis=-1)
    elif points.ndim == 1:
        center_x = (points[0] + points[1]) / 2
        center_y = (points[2] + points[3]) / 2

        width = points[1] - points[0]
        height = points[3] - points[2]
        return np.array([center_x, center_y, width, height])
    else:
        raise ValueError("Available dimensions of points : 1 or 2")


def center2minmax(centers):
    if centers.ndim == 2:
        center_x = centers[:,0]
        center_y = centers[:,1]
        half_width = (np.exp(centers[:,2])-1)/2
        half_height = (np.exp(centers[:,3])-1)/2
        min_x, max_x = center_x - half_width, center_x + half_width
        min_y, max_y = center_y - half_height, center_y + half_height
        return np.stack([min_x, max_x, min_y, max_y],axis=-1)
    elif centers.ndim == 1:
        center_x, center_y = centers[:2]
        half_width = (np.exp(centers[2])-1)/2
        half_height = (np.exp(centers[3])-1)/2
        min_x, max_x = center_x - half_width, center_x + half_width
        min_y, max_y = center_y - half_height, center_y + half_height
        return np.array([min_x, max_x, min_y, max_y])
    else:
        raise ValueError("Available dimensions of centers : 1 or 2")