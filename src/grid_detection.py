#!/usr/bin/env python
# coding: utf-8

import cv2
import numpy as np

from skimage.io import *
from skimage.filters import threshold_otsu
from skimage.morphology import * 
from skimage.transform import *
from skimage.feature import *
from skimage.measure import *
from skimage.draw import line_aa
from skimage.color import rgb2gray
from skimage import img_as_ubyte

def binary_image(image):
    gray = 1 - image
    threshold = threshold_otsu(gray)
    binary = gray > threshold
    return binary > 0

def im_detect_grid(im, dilation_selem=star(3), line_length=1000, line_gap=10):
    im = binary_image(im)
    im = binary_dilation(im, dilation_selem)

    lines = probabilistic_hough_line(im, line_length=line_length, line_gap=line_gap)
    grid_mask = np.zeros_like(im, dtype=np.bool)

    for p0, p1 in lines:
        # ignore diagonal lines
        a = (p1[0]-p0[0]) / ((p1[1]-p0[1]) or float('nan'))
        if abs(a) > 1e-1:
            continue

        rr, cc, val = line_aa(*p0[::-1], *p1[::-1])
        grid_mask[rr, cc] = True

    return grid_mask

def process_image(inputFile, exteriorFile, gridFile):
    im = rgb2gray(imread(inputFile))

    mask_grid = im_detect_grid(im)

    mask_exterior = flood_fill(mask_grid, (0,0), 1)
    mask_exterior ^= mask_grid
    mask_exterior = remove_small_holes(mask_exterior, 500000)

    mask_grid[mask_exterior] = False

    imsave(exteriorFile, img_as_ubyte(mask_exterior))
    imsave(gridFile, img_as_ubyte(mask_grid))
