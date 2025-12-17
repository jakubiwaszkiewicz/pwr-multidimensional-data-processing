import math
import scipy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cbook import flatten
from skimage.data import chelsea, moon
from skimage import transform
from skimage.filters.rank import threshold
from scipy import ndimage

# zad 1

fig, ax = plt.subplots(1, 2, figsize=(10, 10))
cat = chelsea()
cat_monochromatic = np.mean(cat, axis=2)

sobel_x = np.array([[1,0,-1],[2,0,-2],[1,0,-1]])
sobel_y = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])

correlate_x = ndimage.correlate(cat_monochromatic,sobel_x)
correlate = ndimage.correlate(correlate_x,sobel_y)

convolve_x = ndimage.convolve(cat_monochromatic,sobel_x)
convolve = ndimage.convolve(convolve_x,sobel_y)


ax[0].imshow(correlate_x)
ax[1].imshow(convolve)
fig.show()
