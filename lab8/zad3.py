import matplotlib.pyplot as plt
import numpy as np
from skimage.transform import resize
from scipy.ndimage import correlate

zebra = plt.imread("./zebra.jpg")

fig, ax = plt.subplots(1, 3)
zebra_monochromatic = np.mean(zebra, axis=2)

zebra_monochromatic = zebra_monochromatic/(zebra_monochromatic.max()/1.0)

w = [2,6,4,-5,-0,-6]

w_float = [float(i) for i in w]

w_float -= np.mean(w_float)
w_float /= np.max(w_float)

w2d = w_float[:,None] * w_float[None,:]

w2d = np.pad(w2d, 2)

zebra_monochromatic = resize(zebra_monochromatic, (150,250))

sizes = [i for i in range(4,24)]

sums = []

sum_max_val_zebra = 0
sum_min_val_zebra = 0

img_max_zebra = correlate(zebra_monochromatic,w2d)
img_min_zebra = correlate(zebra_monochromatic,w2d)

for idx_i, col_size in enumerate(sizes):
    for idx_j, row_size in enumerate(sizes):
        w2d_resized = resize(w2d, (row_size,col_size))
        w2d_resized -= np.mean(w_float)
        w2d_resized /= np.max(w_float)
        filtered_zebra = correlate(zebra_monochromatic,w2d_resized)
        filtered_zebra = filtered_zebra
        sum = np.sum(filtered_zebra)
        sums.append(sum)
        if idx_i == 0 and idx_j == 0:
            img_max_zebra = filtered_zebra
            img_min_zebra = filtered_zebra
            sum_max_val_zebra = sum
            sum_min_val_zebra = sum
        if sum > sum_max_val_zebra:
            img_max_zebra = filtered_zebra
            sum_max_val_zebra = sum
        if sum < sum_min_val_zebra:
            img_min_zebra = filtered_zebra
            sum_min_val_zebra = sum


sums = np.array(sums)
sums_img = np.reshape(sums,(20,20))

ax[0].imshow(sums_img, cmap='bwr')
ax[1].imshow(img_max_zebra, cmap='bwr')
ax[2].imshow(img_min_zebra, cmap='bwr')

fig.show()
