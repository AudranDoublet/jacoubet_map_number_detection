import skimage
import skimage.io
import skimage.segmentation

from skimage import feature
from skimage import color
from skimage import filters

from skimage.filters import threshold_otsu, sobel
from skimage.measure import label, regionprops
from skimage.feature import peak_local_max

from skimage.morphology import *
from scipy import ndimage as ndi

import numpy as np

import matplotlib.pyplot as plt

from scipy.ndimage.morphology import distance_transform_edt

def show_image(img):
    """
    Show a single image
    """
    fig = plt.figure(figsize = (15,10))
    plt.imshow(img)

def save_image(img, name):
    skimage.io.imsave(name, img)


# In[2]:


def otsu_image(image):
    gray = 1 - image
    threshold = threshold_otsu(gray)

    return binary_dilation(gray > threshold)


# In[3]:


def filter_noise(img, threshold=0.3, m_w=40):
    label_image = label(img)

    for i, region in enumerate(regionprops(label_image)):
        bbox = region.bbox
        h = bbox[2] - bbox[0]
        w = bbox[3] - bbox[1]

        ratio = region.minor_axis_length / region.major_axis_length

        box = label_image[bbox[0]:bbox[2],bbox[1]:bbox[3]]
        
        if ratio < threshold or max(h, w) > m_w or region.extent < threshold or min(w, h) / max(w, h) < threshold:
            sm = img[bbox[0]:bbox[2],bbox[1]:bbox[3]]
            sm[box == i + 1] = 0
            continue

    return img


# In[62]:


def preprocess_image(image, grid, exterior):
    binary = binary_closing(np.any(otsu_image(image), axis=2))
    original = binary.copy()
    binary = binary ^ filter_noise(binary ^ remove_small_objects(binary, 5000), threshold=0.1, m_w=100)
    binary[exterior > 0] = 0
    binary[grid > 0] = 0

    binary = binary ^ filter_noise(binary ^ remove_small_objects(binary, 5000) ^ (binary ^ remove_small_objects(binary, 100)))
    #binary = binary_closing(binary, disk(10))
    binary = remove_small_objects(binary, 100)

    return original, binary


# In[149]:


def dist_map(binary, exterior):
    binary = 1 - binary

    distance = ndi.distance_transform_edt(binary)

    local_maxi = peak_local_max(distance, indices=False, footprint=np.ones((3, 3)), labels=binary)
    markers = ndi.label(local_maxi)[0]
    labels = watershed(-distance, markers, mask=binary)

    labels = labels + 1
    labels[exterior > 0] = 0

    n_dist = distance.copy()

    for i, reg in enumerate(regionprops(labels)):
        bbox = reg.bbox

        label_box = labels[bbox[0]:bbox[2],bbox[1]:bbox[3]]
        dists_box = n_dist[bbox[0]:bbox[2],bbox[1]:bbox[3]]
        
        label_filter = label_box == (i + 1)

        if dists_box[label_filter].shape[0] == 0:
            print("euh")
            continue
        
        h = bbox[2] - bbox[0]
        w = bbox[3] - bbox[1]

        val = np.max(dists_box[label_filter])
        dists_box[label_filter] = val

    return n_dist


# In[6]:


def filter_roads(binary, n_dist, exterior):
    n_binary = binary.copy()
    n_binary[exterior > 0] = 1
    n_binary[n_dist < 10] = 1
    n_binary[n_dist > 100] = 1
    n_binary = remove_small_holes(n_binary, 800000)

    return n_binary


def process_file(inputFile, gridFile, exteriorFile, outputFile, linesOutputFile):
    image = skimage.io.imread(inputFile)
    grid = skimage.io.imread(gridFile) > 0
    exterior = skimage.io.imread(exteriorFile) > 0
    original, binary = preprocess_image(image, grid, exterior)
    n_dist = dist_map(binary, exterior)

    n_binary = filter_roads(binary, n_dist, exterior)

    edges = filters.sobel(n_dist)
    edges[exterior > 0] = 11

    edges = edges > 20
    lines = edges.copy()
    edges[n_dist > 300] = True


    edges = remove_small_holes(edges, 1000)

    v2 = binary.copy()
    v2[edges > 0] = True
    show_image(1 - v2)


    v2[exterior > 0] = 1
    labels = label(1 - v2, connectivity=1)

    for i, reg in enumerate(regionprops(labels)):
        bbox = reg.bbox

        h = bbox[2] - bbox[0]
        w = bbox[3] - bbox[1]

        ratio = reg.minor_axis_length / max(reg.major_axis_length, 1)
        
        if reg.perimeter > 4000 and (reg.perimeter / reg.area) < 0.1:
            continue

        if ratio < 0.1 and reg.minor_axis_length > 20.0:
            continue
            
        if max(w, h) < 2000 and (w + h) < 3000:
            label_box = labels[bbox[0]:bbox[2],bbox[1]:bbox[3]]
            dists_box = v2[bbox[0]:bbox[2],bbox[1]:bbox[3]]
            
            label_filter = label_box == (i + 1)

            dists_box[label_filter] = 1

    skimage.io.imsave(outputFile, v2*255)
    skimage.io.imsave(linesOutputFile, edges*255)
