import numpy as np
from numpy.ma.extras import vstack, dstack
import matplotlib.pyplot as plt
from skimage.data import chelsea, moon
from scipy import ndimage
from scipy.ndimage import median_filter

fig, ax = plt.subplots(3, 3)

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

def median_filter_self(photo, N):
    mean_kernel = (N,N)
    red_mean_cat = median_filter(photo[:,:,0], mean_kernel)
    green_mean_cat = median_filter(photo[:,:,1], mean_kernel)
    blue_mean_cat = median_filter(photo[:,:,2], mean_kernel)
    mean_cat = np.dstack((red_mean_cat,green_mean_cat,blue_mean_cat))
    return mean_cat

ax[0,1].imshow(median_filter_self(noise_p1, 9))
ax[1,1].imshow(median_filter_self(noise_p2, 9))
ax[2,1].imshow(median_filter_self(noise_p3, 9))


def median_filter_vec(data, filter_size):
    temp = []
    indexer = filter_size // 2
    data_final = np.zeros((len(data),len(data[0])))
    for i in range(len(data)):
        for j in range(len(data[0])):
            for z in range(filter_size):
                if i + z - indexer < 0 or i + z - indexer > len(data) - 1:
                    for c in range(filter_size):
                        temp.append(0)
                else:
                    if j + z - indexer < 0 or j + indexer > len(data[0]) - 1:
                        temp.append(0)
                    else:
                        for k in range(filter_size):
                            temp.append(data[i + z - indexer][j + k - indexer])
            temp.sort()
            data_final[i][j] = temp[len(temp) // 2]
            temp = []
    return data_final

r = median_filter_vec(noise_p1[:,:,0], 9)
g = median_filter_vec(noise_p2[:,:,1], 9)
b = median_filter_vec(noise_p2[:,:,2], 9)

img = dstack([r,g,b])

ax[0,2].imshow(img)
ax[1,2].imshow(img)

fig.show()
