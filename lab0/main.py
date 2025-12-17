import numpy as np
import matplotlib.pyplot as plt
def zad1():
    mono = np.zeros((30,30), dtype=np.int32)

    mono[10:20, 10:20] = 1

    mono[15:25, 15:25] = 2

    return mono
def zad2():
    a = zad1()
    fig, ax = plt.subplots(2,2, figsize=(7,7))
    ax[0,0].imshow(a)
    ax[0,1].imshow(a, cmap='binary')

def zad3():
    a = zad1()
    fig, ax = plt.subplots(2, 2, figsize=(7, 7))
    ax[0, 0].imshow(a)
    ax[0, 0].title.set_text('monochromatyczny')
    ax[0, 1].imshow(a, cmap='binary')
    ax[0, 1].title.set_text('monochromatyczny (binary)')
    color = np.zeros((30,30,3))
    color[15:25, 5:15, 0] = 1
    color[10:20, 10:20, 1] = 1
    color[5:15, 15:25, 2] = 1
    ax[1,0].imshow(color)
    ax[1, 0].title.set_text('rgb')
    negative = 1 - color
    ax[1,1].imshow(negative)
    ax[1, 1].title.set_text('cmyk')
    plt.savefig('zad3.png')

zad3()
