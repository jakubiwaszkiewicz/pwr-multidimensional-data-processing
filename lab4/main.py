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

fig, ax = plt.subplots(3, 2)
cat = chelsea()
cat_monochromatic = np.mean(cat, axis=2, dtype=int)

sobel_x = np.array([[1,0,-1],[2,0,-2],[1,0,-1]])

correlate_x = ndimage.correlate(cat_monochromatic,sobel_x)

convolve_x = ndimage.convolve(cat_monochromatic,sobel_x)

ax[0][0].imshow(correlate_x, cmap='Greys')
ax[0][0].title.set_text("Correlate scipy")

ax[0][1].imshow(convolve_x, cmap='Greys')
ax[0][1].title.set_text("Convolve scipy")

def correlation(base_image, kernel):
    convoluted_image = np.zeros_like(base_image)
    if kernel.shape[0] != kernel.shape[1]:
        raise ValueError("kernel must be square")
    kernel_side = kernel.shape[0]
    base_image_width = base_image.shape[0]
    base_image_height = base_image.shape[1]
    padd_with = kernel_side // 2
    for row_index in range(padd_with, base_image_width - padd_with):
        for column_index in range(padd_with, base_image_height - padd_with):
            convoluted_image[row_index, column_index] = (
                    kernel * base_image[
                             row_index - padd_with : row_index + kernel_side - padd_with,
                             column_index - padd_with : column_index + kernel_side - padd_with
                    ]).sum()
    print(f"Base image shape: {base_image.shape}")
    print(f"Convoluted image shape: {convoluted_image.shape}")
    return convoluted_image


hand_made_correlation = correlation(cat_monochromatic, sobel_x)
conv_sobel_x = np.flip(sobel_x)
hand_made_convolution = correlation(cat_monochromatic, conv_sobel_x)


ax[1][0].imshow(hand_made_correlation, cmap='Greys')
ax[1][0].title.set_text("Handmade corr")

ax[1][1].imshow(hand_made_convolution, cmap='Greys')
ax[1][1].title.set_text("Handmade conv")



ax[2][0].imshow(abs(hand_made_correlation-correlate_x))
sum_corr = sum(flatten(abs(hand_made_correlation-correlate_x)))
ax[2][0].title.set_text(f"Różnica: {sum_corr}")
ax[2][1].imshow(abs(hand_made_convolution-convolve_x))
sum_conv = sum(flatten(abs(hand_made_convolution-convolve_x)))
ax[2][1].title.set_text(f"Różnica {sum_conv}")


# fig.show()


fig2, ax2 = plt.subplots(2, 2)
cat = chelsea()
cat_monochromatic = np.mean(cat, axis=2, dtype=int)

ax2[0][0].imshow(cat_monochromatic, vmin=0, vmax=255)

blur = np.ones((7,7))
blur /= np.sum(blur)

blured_cat = correlation(cat_monochromatic, blur)

ax2[0][1].imshow(blured_cat, vmin=0, vmax=255)

mask = cat_monochromatic - blured_cat

ax2[1][0].imshow(mask)

ax2[1][1].imshow(cat_monochromatic + mask, vmin=0, vmax=255)

fig2.show()