import matplotlib.pyplot as plt
import numpy as np
from skimage.transform import resize
from scipy.ndimage import correlate

zebra = plt.imread("./zebra.jpg")

fig, ax = plt.subplots(3, 2)
zebra_monochromatic = np.mean(zebra, axis=2)

zebra_monochromatic = zebra_monochromatic/(zebra_monochromatic.max()/1.0)

w = [2,6,4,-5,-0,-6]

w_float = [float(i) for i in w]

w_float -= np.mean(w_float)
w_float /= np.max(w_float)

w2d = w_float[:,None] * w_float[None,:]

w2d = np.pad(w2d, 2)

w2d30x30 = resize(w2d, (30,30))


ax[0,0].plot(w)
ax[0,1].plot(w_float)
ax[1,0].imshow(w2d, cmap='bwr')
ax[1,1].imshow(w2d30x30, cmap='bwr')
ax[2,0].imshow(zebra_monochromatic, cmap='bwr')
ax[2,1].imshow(correlate(zebra_monochromatic, w2d30x30), cmap='bwr')

fig.show()