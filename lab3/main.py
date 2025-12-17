import math
import scipy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cbook import flatten
from skimage.data import chelsea, moon
from skimage import transform
from skimage.filters.rank import threshold


def zad1():
    fig, ax = plt.subplots(6, 3)
    cat = chelsea()
    d = 8
    l = pow(2,d)
    base_fn = np.linspace(0, l-1, l)
    base_fn = base_fn.astype(dtype=int)
    ax[0,0].plot(base_fn, base_fn)
    ax[0,1].imshow(base_fn[cat])

    negative_base_fn =- base_fn + l-1
    ax[1,0].scatter(base_fn, negative_base_fn)
    ax[1,1].imshow(negative_base_fn[cat])

    threshold_temp_base_fn = [[0 for _ in range(50)], [1 for _ in range(150 - 50)], [0 for _ in range(l - 150)]]
    threshold_base_fn_list = []
    for fn in threshold_temp_base_fn:
        threshold_base_fn_list.extend(fn)
    threshold_base_fn = np.array(threshold_base_fn_list) * l
    ax[2,0].scatter(base_fn, threshold_base_fn)
    ax[2,1].imshow(threshold_base_fn[cat])

    sin_base_fn = np.sin(np.linspace(0, 2*np.pi, l))
    sin_base_fn-= np.min(sin_base_fn)
    sin_base_fn/= np.max(sin_base_fn)
    sin_base_fn*= 255
    sin_base_fn = sin_base_fn.astype(int)
    ax[3,0].scatter(base_fn, sin_base_fn)
    ax[3,1].imshow(sin_base_fn[cat])


    gamma_0_3_base_fn = pow(base_fn, 0.3)
    gamma_0_3_base_fn-= np.min(gamma_0_3_base_fn)
    gamma_0_3_base_fn/= np.max(gamma_0_3_base_fn)
    gamma_0_3_base_fn*= 255
    gamma_0_3_base_fn = gamma_0_3_base_fn.astype(int)
    ax[4,0].scatter(base_fn, gamma_0_3_base_fn)
    ax[4,1].imshow(gamma_0_3_base_fn[cat])

    gamma_3_base_fn = pow(base_fn, 3)
    gamma_3_base_fn = gamma_3_base_fn.astype(float)
    gamma_3_base_fn -= np.min(gamma_3_base_fn)
    gamma_3_base_fn /= np.max(gamma_3_base_fn)
    gamma_3_base_fn *= 255
    gamma_3_base_fn = gamma_3_base_fn.astype(int)
    ax[5,0].scatter(base_fn, gamma_3_base_fn)
    ax[5,1].imshow(gamma_3_base_fn[cat])

    def rgb_print(fns):
        for idx, fn in enumerate(fns):
            image = fn[cat]
            values, counts = np.unique(image[:, :, 0], return_counts=True)
            counts_arr = counts / np.sum(counts)
            ax[idx, 2].scatter(values, counts_arr, color='red', s=1)

            values, counts = np.unique(image[:, :, 1], return_counts=True)
            counts_arr = counts / np.sum(counts)
            ax[idx, 2].scatter(values, counts_arr, color='green', s=1)

            values, counts = np.unique(image[:, :, 2], return_counts=True)
            counts_arr = counts / np.sum(counts)
            ax[idx, 2].scatter(values, counts_arr, color='blue', s=1)
    rgb_print([base_fn, negative_base_fn, threshold_base_fn, sin_base_fn, gamma_0_3_base_fn,gamma_3_base_fn])

    plt.show()
def zad2():
    fig, ax = plt.subplots(2, 3)
    space = moon()
    ax[0, 0].imshow(space)

    values, counts = np.unique(space, return_counts=True)

    ax[0, 1].bar(values, counts)

    dyst = np.cumsum(counts)
    dyst_int = dyst.astype(int)
    dyst_int -= np.min(dyst_int)
    dyst_int /= np.max(dyst_int)
    dyst_arr = np.linspace(0,255,dyst_int.shape[0])
    ax[0,2].scatter(dyst_arr, dyst)

    lut = dyst * 255
    ax[1, 0].scatter(np.arange(lut.shape[0]), lut)

    ax[1, 1].imshow(lut[moon], cmap='binary_r')

    values, counts = np.unique(lut[moon], return_counts=True)
    counts_arr = counts / np.sum(counts)
    ax[1, 2].bar(values, counts_arr)

    plt.show()
zad2()