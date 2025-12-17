import numpy as np
from numpy.ma.extras import vstack
import matplotlib.pyplot as plt
from skimage.data import chelsea, moon
from scipy import ndimage

fig, ax = plt.subplots(1, 3)

cat = chelsea()


reduced_cat = cat[::3,::3]

ax[0].imshow(reduced_cat)

red = reduced_cat[:,:,0]
green = reduced_cat[:,:,1]
blue = reduced_cat[:,:,2]

possible_values = [True, False]
p = [0.2,0.8]

where_to_change = np.random.choice(possible_values, size=reduced_cat.shape, p=p)

noise = reduced_cat

noise[where_to_change] = np.random.choice([0,255])

ax[1].imshow(noise)


N=3
mean_kernel = np.ones((N,N))/(N**2)

red_mean_cat = ndimage.correlate(red,mean_kernel)
green_mean_cat = ndimage.correlate(green,mean_kernel)
blue_mean_cat = ndimage.correlate(blue,mean_kernel)

mean_cat = np.dstack((red_mean_cat,green_mean_cat,blue_mean_cat))

ax[2].imshow(mean_cat)


fig.show()
