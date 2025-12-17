import matplotlib.pyplot as plt
import numpy as np
from skimage.transform import resize
from scipy.ndimage import correlate

zebra = plt.imread("./zebra.jpg")

fig, ax = plt.subplots(6, 6)
zebra_monochromatic = np.mean(zebra, axis=2)

zebra_monochromatic = zebra_monochromatic/(zebra_monochromatic.max()/1.0)

w = [2,6,4,-5,-0,-6]

w_float = [float(i) for i in w]

w_float -= np.mean(w_float)
w_float /= np.max(w_float)

w2d = w_float[:,None] * w_float[None,:]

w2d = np.pad(w2d, 2)

sizes = [4,9,15,20,26,32]

falka = []

for idx_i, col_size in enumerate(sizes):
    for idx_j, row_size in enumerate(sizes):
        w2d_resized = resize(w2d, (row_size,col_size))
        w2d_resized -= np.mean(w_float)
        w2d_resized /= np.max(w_float)
        ax[idx_i,idx_j].imshow(correlate(zebra_monochromatic,w2d_resized), cmap='bwr')


fig.show()
