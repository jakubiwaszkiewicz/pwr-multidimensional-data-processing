import math
import scipy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cbook import flatten
from skimage.data import chelsea
from skimage import transform


def zad1():
    fig, ax = plt.subplots(3, 2, figsize=(10, 7))
    cat = chelsea()
    ax[0, 0].imshow(cat, cmap="magma")

    cat_monochromatic = np.mean(cat, axis=2)
    reduced_cat_monochromatic = cat_monochromatic[::8,::8]

    ax[0, 1].imshow(reduced_cat_monochromatic, cmap="binary_r")

    rad_value = math.pi/12
    first_row = [math.cos(rad_value) ,-math.sin(rad_value),0]
    second_row = [math.sin(rad_value) ,math.cos(rad_value),0]
    third_row = [0,0,1]

    matrix_rotate = [first_row, second_row, third_row]

    matrix_shear_first_row = [1, 0.5, 0]
    matrix_shear_second_row = [0 , 1 ,0]
    matrix_shear_third_row = [0 , 0 , 1]

    matrix_shear = [matrix_shear_first_row, matrix_shear_second_row, matrix_shear_third_row]

    at = transform.AffineTransform(matrix_rotate)  # tworzy transformację afiniczną
    transformed_cat = transform.warp(cat, inverse_map=at.inverse)  # przekształca obraz według transformacji
    ax[1, 0].imshow(transformed_cat, cmap="binary_r")

    at_shear = transform.AffineTransform(matrix_shear)
    transformed_shear_cat = transform.warp(cat, inverse_map=at_shear.inverse)
    ax[1, 1].imshow(transformed_shear_cat, cmap="binary_r")
    return reduced_cat_monochromatic

def zad2(reduced_cat_monochromatic):
    fig, ax = plt.subplots(1, 2, figsize=(9, 6))
    arr = []
    for idr, row in enumerate(reduced_cat_monochromatic):
        for idp, _ in enumerate(row):
            arr.append([idr,idp])
    arr = np.array(arr)
    flatten_image = reduced_cat_monochromatic.flatten()

    ax[0].scatter(arr[:,0], arr[:,1], c=flatten_image, cmap='binary_r')

    arr = np.column_stack((arr, np.ones(len(arr))))

    rad_value = math.pi/12
    first_row = [math.cos(rad_value) ,-math.sin(rad_value),0]
    second_row = [math.sin(rad_value) ,math.cos(rad_value),0]
    third_row = [0,0,1]

    matrix_rotate = [first_row, second_row, third_row]

    arr_rotated = arr @ matrix_rotate

    ax[1].scatter(arr_rotated[:,0], arr_rotated[:,1], c=flatten_image, cmap='binary_r')

def zad3(reduced_cat_monochromatic):
    fig, ax = plt.subplots(1, 3, figsize=(9, 6))
    position = []
    for idr, row in enumerate(reduced_cat_monochromatic):
        for idp, _ in enumerate(row):
            position.append([idr,idp])
    position = np.array(position)
    intensity = reduced_cat_monochromatic.flatten()
    ax[0].scatter(position[:,0], position[:,1], c=intensity, cmap='binary_r')
    position_with_intensity = np.column_stack((position, intensity))

    random_indexes = np.random.choice(len(position_with_intensity), 1000)

    random_pos_intensity = position_with_intensity[random_indexes]

    ax[1].scatter(random_pos_intensity[:,0], random_pos_intensity[:,1], c=random_pos_intensity[:,2], cmap='binary_r')

    interpolation = []

    for i in range(300):
        for j in range(400):
            interpolation.append([i, j])

    interpolation_dist = scipy.spatial.distance.cdist(random_pos_intensity[:,2], interpolation)

    ax[2].scatter(interpolation_dist[:, 0], interpolation_dist[:, 1], c=random_pos_intensity[:,2], cmap='binary_r')

reduced_cat_monochromatic = zad1()

zad3(reduced_cat_monochromatic)

plt.show()
