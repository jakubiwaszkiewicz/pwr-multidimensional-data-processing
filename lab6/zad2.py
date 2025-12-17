import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
hsi_small = np.load('./hsi_small.npy')

print(hsi_small.shape)

fig, ax = plt.subplots(1, 2)


r = hsi_small[:,:,100:101]
g = hsi_small[:,:,50:51]
b = hsi_small[:,:,7:8]


r -= np.min(r)
r /= np.max(r)

g -= np.min(g)
g /= np.max(g)

b -= np.min(b)
b /= np.max(b)

rgb = np.dstack((r,g,b))

ax[0].imshow(rgb)

prepared_hsi = hsi_small.reshape(65536, 204)

pca = PCA(n_components=3)
pca_transformed_hsi = pca.fit_transform(prepared_hsi)

img_pca = pca_transformed_hsi.reshape(256,256,3)

r = img_pca[:,:,0]
g = img_pca[:,:,1]
b = img_pca[:,:,2]

r -= np.min(r)
r /= np.max(r)

g -= np.min(g)
g /= np.max(g)

b -= np.min(b)
b /= np.max(b)

p = np.dstack((r,g,b))

ax[1].imshow(p)

fig.show()
