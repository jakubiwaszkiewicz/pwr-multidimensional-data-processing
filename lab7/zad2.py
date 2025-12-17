import numpy as np
from numpy.ma.extras import vstack
import matplotlib.pyplot as plt
from skimage.data import chelsea, moon
from scipy import ndimage
from scipy.ndimage import median_filter

fig, ax = plt.subplots(3, 4)

cat = chelsea()


reduced_cat = cat[::3,::3]

possible_values = [True, False]

p1 = [0.1, 0.9]
p2 = [0.3, 0.7]
p3 = [0.45, 0.55]

where_to_change_p1 = np.random.choice(possible_values, size=reduced_cat.shape, p=p1)
where_to_change_p2 = np.random.choice(possible_values, size=reduced_cat.shape, p=p2)
where_to_change_p3 = np.random.choice(possible_values, size=reduced_cat.shape, p=p3)



noise_p1 = np.copy(reduced_cat)
noise_p2 = np.copy(reduced_cat)
noise_p3 = np.copy(reduced_cat)

noise_p1[where_to_change_p1] = np.random.choice([0,255])
noise_p2[where_to_change_p2] = np.random.choice([0,255])
noise_p3[where_to_change_p3] = np.random.choice([0,255])

ax[0,0].imshow(noise_p1)
ax[1,0].imshow(noise_p2)
ax[2,0].imshow(noise_p3)

def mean_filter(photo, N):
    mean_kernel = (N,N)

    red_mean_cat = median_filter(photo[:,:,0], mean_kernel)
    green_mean_cat = median_filter(photo[:,:,1], mean_kernel)
    blue_mean_cat = median_filter(photo[:,:,2], mean_kernel)

    mean_cat = np.dstack((red_mean_cat,green_mean_cat,blue_mean_cat))

    return mean_cat

ax[0,1].imshow(mean_filter(noise_p1, 3))
ax[1,1].imshow(mean_filter(noise_p2, 3))
ax[2,1].imshow(mean_filter(noise_p3, 3))


ax[0,2].imshow(mean_filter(noise_p1, 5))
ax[1,2].imshow(mean_filter(noise_p2, 5))
ax[2,2].imshow(mean_filter(noise_p3, 5))


ax[0,3].imshow(mean_filter(noise_p1, 9))
ax[1,3].imshow(mean_filter(noise_p2, 9))
ax[2,3].imshow(mean_filter(noise_p3, 9))
fig.show()
