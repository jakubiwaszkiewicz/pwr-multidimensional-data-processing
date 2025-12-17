import numpy as np
import matplotlib.pyplot as plt
from numpy.ma.extras import vstack

hsi_small = np.load('./hsi_small.npy')

fig, ax = plt.subplots(2, 3)

X = [i for i in range(204)]
print(len(X))

ax[0,0].imshow(hsi_small[:,:,10:11])
ax[0,0].scatter(x=25,y=25, c="r")
ax[0,1].imshow(hsi_small[:,:,100:101])
ax[0,1].scatter(x=200,y=50, c="r")
ax[0,2].imshow(hsi_small[:,:,200:201])
ax[0,2].scatter(x=150,y=175, c="r")

print(len(hsi_small[25:26,25:26,:].flatten()))

ax[1,0].plot(hsi_small[25:26,25:26,:].flatten(), c="b")
ax[1,0].scatter(x=X[10],y=hsi_small[25:26,25:26,10:11], c="r")


ax[1,1].plot(hsi_small[200:201, 50:51,:].flatten(), c="b")
ax[1,1].scatter(x=X[100],y=hsi_small[200:201, 50:51,100:101], c="r")


ax[1,2].plot(hsi_small[150:151, 175:176,:].flatten(), c="b")
ax[1,2].scatter(x=X[200],y=hsi_small[150:151, 175:176, 200:201], c="r")

fig.show()


###zad2

r = hsi_small[:,:,100:101].flatten()
g = hsi_small[:,:,50:51].flatten()
b = hsi_small[:,:,7:8].flatten()

rgb = np.dstack((r,g,b))





