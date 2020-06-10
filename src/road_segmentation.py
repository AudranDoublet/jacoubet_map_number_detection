import skimage
import skimage.io
import skimage.segmentation

from skimage import feature
from skimage import color

from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops

from skimage.morphology import *

import numpy as np

import matplotlib.pyplot as plt

from scipy.ndimage.morphology import distance_transform_edt

import cv2

def otsu_image(image, threshold=0):
    gray = 1 - image
    threshold = threshold_otsu(gray)

    return binary_dilation(gray > threshold)


def process_file(inputFile, gridFile, outputFile):
    image = skimage.io.imread(inputFile)
    road_mask = skimage.io.imread(gridFile) > 120
    binary = remove_small_objects(np.any(otsu_image(image), axis=2), 5000)

    img = binary

    img = ((255*img) - (road_mask*255)) > 0
    img = binary_closing(img, square(15))
    img = remove_small_holes(img, 10000000)

    skimage.io.imsave(outputFile, img*255)
