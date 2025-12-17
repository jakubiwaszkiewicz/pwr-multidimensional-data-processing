import numpy as np
import matplotlib.pyplot as plt

fig, ax = plt.subplots(3, 3, figsize=(10, 10))

def zad1(ax):
    x = np.linspace(0, np.pi*4, 40)
    y = np.sin(x)
    ax[0, 0].plot(x,y)
    y_2d = y[:, np.newaxis] * y[np.newaxis, :]

    ax[0, 1].imshow(y_2d, cmap="magma")
    ax[0, 1].title.set_text(f"Min: {np.round(np.min(y), decimals=3)}, Max: {np.round(np.max(y), decimals=3)}")

    y_2d_normalized = (y_2d - np.min(y_2d)) / (np.max(y_2d) - np.min(y_2d))
    ax[0, 2].imshow(y_2d_normalized, cmap="magma")
    return y_2d_normalized

def zad2(ax, y_2d_normalized):

    y_2d_2bit = np.rint(y_2d_normalized * np.power(2, 2))
    ax[1, 0].imshow(y_2d_2bit, cmap="magma")
    ax[1, 0].title.set_text(f"Min: {np.round(np.min(y_2d_2bit), decimals=3)}, Max: {np.round(np.max(y_2d_2bit), decimals=3)}")

    y_2d_4bit = np.rint(y_2d_normalized * np.power(2, 4))
    ax[1, 1].imshow(y_2d_4bit, cmap="magma")
    ax[1, 1].title.set_text(f"Min: {np.round(np.min(y_2d_4bit), decimals=3)}, Max: {np.round(np.max(y_2d_4bit), decimals=3)}")

    y_2d_8bit = np.rint(y_2d_normalized * np.power(2,8))
    ax[1, 2].imshow(y_2d_8bit, cmap="magma")
    ax[1, 2].title.set_text(f"Min: {np.round(np.min(y_2d_8bit), decimals=3)}, Max: {np.round(np.max(y_2d_8bit), decimals=3)}")

def zad3(ax, y_2d_normalized):
    noise = np.random.normal(size=y_2d_normalized.shape)
    noised_y_2d_normalized = y_2d_normalized + noise
    ax[2, 0].imshow(noised_y_2d_normalized, cmap="magma")
    ax[2, 0].title.set_text(f"n=1")
    noised_y_2d_50 = [ y_2d_normalized + np.random.normal(size=y_2d_normalized.shape) for _ in range(50) ]
    noised_y_2d_1000 = [ y_2d_normalized + np.random.normal(size=y_2d_normalized.shape) for _ in range(1000) ]
    noised_y_2d_50_mean = np.mean(noised_y_2d_50, axis=0)
    noised_y_2d_1000_mean = np.mean(noised_y_2d_1000, axis=0)
    ax[2, 1].imshow(noised_y_2d_50_mean, cmap="magma")
    ax[2, 1].title.set_text(f"n=50")
    ax[2, 2].imshow(noised_y_2d_1000_mean, cmap="magma")
    ax[2, 2].title.set_text(f"n=1000")

y_2d_normalized = zad1(ax)
zad2(ax, y_2d_normalized)
zad3(ax, y_2d_normalized)
fig.show()