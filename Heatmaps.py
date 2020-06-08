#!/usr/bin/env python
# coding: utf-8

# In[16]:


import numpy as np
import cv2
import os

import skimage
import skimage.io

from skimage.filters import threshold_otsu, threshold_sauvola, inverse
from skimage.color import rgb2gray
from skimage.measure import label, regionprops

import matplotlib.pyplot as plt

from skimage.morphology import *


# In[2]:


FILES = get_ipython().getoutput('ls inputs/*')


# In[3]:


#image = skimage.io.imread(FILES[-1])
image = skimage.io.imread(FILES[3])[5000:6200, 200: 2000, :]
#image = skimage.io.imread(FILES[-1])[6200:7700, 4600: 6100, :]
# image = skimage.io.imread(FILES[2])[4000:5000, 2000:3000, :]

plt.figure(figsize=(16,16))
imgplot = plt.imshow(image,  cmap=plt.cm.gray)


# In[4]:


denoised = rgb2gray(cv2.fastNlMeansDenoising(image))


# In[5]:


plt.figure(figsize=(16,16))
imgplot = plt.imshow(denoised,  cmap=plt.cm.gray)


# In[6]:


def otsu_image(image):
    gray = 1 - image
    threshold = threshold_otsu(gray)

    return binary_dilation(gray > threshold)


# In[7]:


binary_otsu = otsu_image(denoised)

plt.figure(figsize=(16,16))
imgplot = plt.imshow(binary_otsu, cmap=plt.cm.gray)


# In[9]:


def binary_image(image):
    gray = 1 - rgb2gray(image)

    threshold = threshold_sauvola(gray, window_size=25, k=0.05)
    binary = gray > threshold
    binary = binary * 255 - remove_small_objects(binary, 400) * 255
    return binary > 0

binary = binary_image(denoised)

plt.figure(figsize=(16,16))
imgplot = plt.imshow(binary,  cmap=plt.cm.gray)


# In[35]:


# conserve only objects of the size of a number
small = (remove_small_objects(binary, 5) * 255) - (remove_small_objects(binary, 400) * 255)

plt.figure(figsize=(16,16))
imgplot = plt.imshow(small,  cmap=plt.cm.gray)


# In[38]:


def filter_results_simple(small):
    label_image = label(small)

    for i, region in enumerate(regionprops(label_image)):
        bbox = region.bbox
        ratio = region.minor_axis_length / region.major_axis_length
        box = label_image[bbox[0]:bbox[2],bbox[1]:bbox[3]]

        if ratio < 0.25 or region.minor_axis_length < 4:
            sm = small[bbox[0]:bbox[2],bbox[1]:bbox[3]]
            sm[box == i + 1] = 0
            continue

        lbl = denoised[bbox[0]:bbox[2],bbox[1]:bbox[3]][box == i + 1]

        total_count = lbl.shape[0]
        black_count = np.sum(lbl <= 0.5)# - total_count / 3

        total_square_count = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        active_square_count = np.sum(box == i + 1)

        if black_count / total_count < 0.1 or active_square_count / total_square_count < 0.2:
            sm = small[bbox[0]:bbox[2],bbox[1]:bbox[3]]
            sm[box == i + 1] = 0
            continue

    return small

small = filter_results_simple(small)


# In[39]:


plt.figure(figsize=(16,16))
imgplot = plt.imshow(small,  cmap=plt.cm.gray)


# In[20]:


def filter_results_connex(small):
    closed_small = dilation(small, square(30))

    im = (binary_otsu*255 - small) > 0
    closed_small[im > 0] = False
    closed_small = closed_small > 0
    closed_small = remove_small_holes(closed_small, 1000)

    closed_small = remove_small_objects(closed_small, 1000)*255 - remove_small_objects(closed_small, 10000)*255

    label_image = label(closed_small)

    for i, region in enumerate(regionprops(label_image)):
        bbox = region.bbox
        box = label_image[bbox[0]:bbox[2],bbox[1]:bbox[3]]

        ratio = region.minor_axis_length / region.major_axis_length

        if ratio < 0.4:
            sm = closed_small[bbox[0]:bbox[2],bbox[1]:bbox[3]]
            sm[box == i + 1] = 0
    
    return closed_small

closed_small = filter_results_connex(small)


# In[21]:


result = np.bitwise_and(closed_small > 0, small > 0)
plt.figure(figsize=(16,16))
imgplot = plt.imshow(result, cmap=plt.cm.gray)

