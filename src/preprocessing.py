import skimage
from skimage.io import imread, imsave

from skimage import feature
from skimage import color
from skimage import filters

from skimage.morphology import *

import numpy as np

def otsu_image(image):
    gray = 1 - image
    threshold = filters.threshold_otsu(gray.reshape(-1, gray.shape[1], 1))

    return binary_dilation(gray > threshold)


def get_striped_house_candidates(exterior, binary):
    closed = binary_closing(binary, disk(3))
    closed = binary_opening(closed, disk(15))
    closed[exterior > 0] = False

    return remove_small_objects(closed, 1024)


def clean_striped_houses(image, exterior):
    binary = np.any(otsu_image(image), axis=2)
    candidates = get_striped_house_candidates(exterior, binary)

    gray = color.rgb2gray(image)
    gray[candidates < 1] = 0

    blurred = filters.gaussian(gray, sigma=2)
    mask = remove_small_objects(blurred < 0.65, 10) | np.bitwise_not(candidates)

    image[mask < 1,:] = np.mean(image.reshape(-1, 3), axis=0)
    return image


def process_file(inputFile, exteriorFile, outputFile):
    image = imread(inputFile)
    exterior = imread(exteriorFile)

    # apply preprocessing
    image = clean_striped_houses(image, exterior)

    imsave(outputFile, image)
