import numpy as np
import cv2
import os

from skimage.io import *
from skimage.color import *
from skimage.filters import *
from skimage.morphology import * 
from skimage.transform import *
from skimage.feature import *
from skimage.measure import *
from skimage.draw import line_aa
from skimage import data
from scipy.ndimage.morphology import distance_transform_edt

def binary_image(image):
    gray = 1 - image
    threshold = threshold_otsu(gray)
    binary = gray > threshold
    return binary > 0


def im_dilate(im, gaussian_sigma=2):
    im = gaussian(im, gaussian_sigma)
    im = binary_image(im)
    im = binary_dilation(im, disk(5))
    return im

def area_ecc_filter(im, area_bounds, ecc_bounds, ext_bounds=(0.0, 1.0)):
    # Extract the region props of the objects. 
    props = regionprops(im)

    # Extract the areas and labels.
    areas = np.array([prop.area for prop in props])
    eccs = np.array([prop.eccentricity for prop in props])
    labels = np.array([prop.label for prop in props])
    extents = np.array([prop.extent for prop in props])

    # Make an empty image to add the approved cells.
    im_approved = np.zeros_like(im)

    # Threshold the objects based on area and eccentricity
    for i, _ in enumerate(areas):
        if areas[i] > area_bounds[0] and areas[i] < area_bounds[1]\
            and eccs[i] > ecc_bounds[0] and eccs[i] < ecc_bounds[1]\
            and extents[i] > ext_bounds[0] and extents[i] < ext_bounds[1]:
                im_approved += im==labels[i]

    return im_approved > 0


class HeatmapWorker:
    def __init__(self, inputFile, roadFile, gridFile, thinGridFile, exteriorFile, outputFile):
        self.image = imread(inputFile)
        self.road_mask = imread(roadFile) > 0
        self.grid_mask = imread(gridFile) > 0
        self.thin_grid_mask = imread(thinGridFile) > 0
        self.exterior_mask = imread(exteriorFile) > 0
        self.output_file = outputFile

 
    def im_detect_lines(self, im, gaussian_sigma=2.5, small_object_size=1024):
        im = gaussian(im, gaussian_sigma)
        im = canny(im)
        im = dilation(im)
        im = remove_small_objects(im > 0.1, small_object_size)
        return im


    def im_detect_lines_area(self, lines, valid_area=50):
        lines = 1-lines
        im = -distance_transform_edt(lines)
        return im > -valid_area


    def im_detect_blobs(self, im):
        im = rgb2gray(im)

        # remove the grid from the image
        mask_grid = self.thin_grid_mask # FIXME dilate ?
        im[mask_grid] = im.max()

        # remove exterior
        im[self.exterior_mask] = im.max()

        # detect lines
        mask_lines = self.im_detect_lines(im, 2.3)
        mask_available_area = self.im_detect_lines_area(mask_lines, 45)

        # dilate and cut dilated blobs according to the previously detected bounding lines
        im = im_dilate(im)
        im &= mask_available_area
        im &= ~mask_lines

        # filter non-viable blobs
        im = erosion(im, star(3))
        im = area_ecc_filter(label(im), (50, 1000), (0., 1.0), (0.3, 1.0))

        return im


    def road_distance_mask(self):
        return distance_transform_edt(1 - self.road_mask)


    def im_remove_objects_by_road_distance(self, im, min_distance=1+1e-3, max_distance=20):
        im = im.copy()
        distance_map = self.road_distance_mask()

        label_image = label(im)

        for reg in regionprops(label_image):
            sh = reg.coords[:,0]
            sw = reg.coords[:,1]

            road_distance = np.min(distance_map[sh, sw])

            # if object is too far or too near from the road, ignore it
            if road_distance > max_distance or road_distance < min_distance:
                im[sh, sw] = 0
                continue

        return im


    def im_remove_small_aligned_objects(self, im, line_length=200, line_gap=25, threshold_size=128):
        # FIXME: - it'd be better to filter the image to have only small objects, 
        #          so we can decrease `line_length`, and increase `line_gap`
        # FIXME-END
        im = im.copy() # that's slow and useless, but it avoids modifying the image given by reference

        lines = probabilistic_hough_line(im, line_length=line_length, line_gap=line_gap)
        mask_lines = np.zeros_like(im, dtype=np.bool)

        # we draw a line joining the small objects of the image
        for p0, p1 in lines:
            rr, cc, val = line_aa(*p0[::-1], *p1[::-1])
            mask_lines[rr, cc] = True

        im[mask_lines] = 0
        return remove_small_objects(im > 0, threshold_size)


    def process_image(self):
        im = self.im_detect_blobs(self.image)
        im = self.im_remove_objects_by_road_distance(im)
        im = self.im_remove_small_aligned_objects(im)

        im = (im * 255).astype(np.uint8)
        imsave(self.output_file, im, check_contrast=False)

        return im


def process_file(*kargs):
    worker = HeatmapWorker(*kargs)
    worker.process_image()
