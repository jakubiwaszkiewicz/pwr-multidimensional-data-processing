import numpy as np
import matplotlib.pyplot as plt
from skimage.data import chelsea, moon

def zad1():
    fig, ax = plt.subplots(2, 3)

    mono_image = np.zeros((1000,1000), dtype=int)

    mono_image [500:520, 460:550] = 1

    ax[0, 0].imshow(mono_image, cmap='magma')

    fft_mono_image = np.fft.fft2(mono_image)

    shifted_fft_mono_image = np.fft.fftshift(fft_mono_image)
    ax[0, 1].imshow(np.log(np.abs(shifted_fft_mono_image.real) + 1), cmap='magma')
    ax[0, 2].imshow(np.log(np.abs(shifted_fft_mono_image.imag) + 1), cmap='magma')

    ax[1, 0].imshow(np.arctan2(shifted_fft_mono_image.imag , shifted_fft_mono_image.real), cmap='magma')

    ax[1, 1].imshow(np.log(np.abs(shifted_fft_mono_image) + 1), cmap='magma')

    shifted_ifft_mono_image = np.fft.ifft2(np.fft.ifftshift(shifted_fft_mono_image))

    ax[1, 2].imshow(shifted_ifft_mono_image.real, cmap='magma')

    plt.show()
#zad1()



def zad2(mono_image):
    fig, ax = plt.subplots(2, 3)

    ax[0, 0].imshow(mono_image, cmap='magma')

    fft_mono_image = np.fft.fft2(mono_image)

    shifted_fft_mono_image = np.fft.fftshift(fft_mono_image)
    ax[0, 1].imshow(np.log(np.abs(shifted_fft_mono_image.real) + 1), cmap='magma')
    ax[0, 2].imshow(np.log(np.abs(shifted_fft_mono_image.imag) + 1), cmap='magma')

    ax[1, 0].imshow(np.arctan2(shifted_fft_mono_image.imag , shifted_fft_mono_image.real), cmap='magma')

    ax[1, 1].imshow(np.log(np.abs(shifted_fft_mono_image) + 1), cmap='magma')

    shifted_ifft_mono_image = np.fft.ifft2(np.fft.ifftshift(shifted_fft_mono_image))

    ax[1, 2].imshow(shifted_ifft_mono_image.real, cmap='magma')

    plt.show()


x_values = np.linspace(1,15, 1000)
y_values = np.linspace(1,15, 1000)

xx, yy = np.meshgrid(x_values, y_values)

rand_amp = [2, 4, 6, 8, 10]
rand_angle = [0.9*np.pi, 2.3*np.pi, 3.7*np.pi, 4.1*np.pi, 5.2*np.pi]
rand_wave_len = [2, 4, 6, 8, 10]

matrix = np.zeros((1000, 1000))

for amp, angle, wave_len in zip(rand_amp, rand_angle, rand_wave_len):
    matrix += amp * np.sin(2 * np.pi * (xx + np.cos(angle) + yy * np.sin(angle)) * 1 / wave_len)

# zad2(matrix)



fig, ax = plt.subplots(2, 3)

cat = chelsea()
cat_monochromatic = np.mean(cat, axis=2)
ax[0, 0].imshow(cat_monochromatic, cmap="Grays")
fft_mono_image = np.fft.fft2(cat_monochromatic)
shifted_fft_mono_image = np.fft.fftshift(fft_mono_image)
ax[0, 1].imshow(np.log(np.abs(shifted_fft_mono_image) + 1))

shifted_ifft_mono_image_real = np.fft.ifft2(np.fft.ifftshift(shifted_fft_mono_image.real))
shifted_ifft_mono_image_imag = np.fft.ifft2(np.fft.ifftshift(shifted_fft_mono_image.imag))
shifted_ifft_mono_image = np.fft.ifft2(np.fft.ifftshift(shifted_fft_mono_image))

r = shifted_ifft_mono_image_real.real
g = shifted_ifft_mono_image_imag.imag
b = shifted_ifft_mono_image.real

ax[1,0].imshow(r, cmap='Reds_r')
ax[1,1].imshow(g, cmap='Greens_r')
ax[1,2].imshow(b, cmap='Blues_r')

r -= np.min(r)
r /= np.max(r)

g -= np.min(g)
g /= np.max(g)

b -= np.min(b)
b /= np.max(b)

rgb = np.dstack((r,g,b))

ax[0,2].imshow(rgb)

plt.show()
