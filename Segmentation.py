#!/usr/bin/env python
# coding: utf-8

# # Utils functions

# In[1]:


import skimage.io
def load_image(image_path):
    """
    Load image with its path
    """
    return skimage.io.imread(image_path)


# In[2]:


import matplotlib.pyplot as plt
plt.rcParams['image.cmap'] = 'gray'
def show_image(img):
    """
    Show a single image
    """
    fig = plt.figure(figsize = (15,10))
    plt.imshow(img)


# In[3]:


from skimage.measure import label, regionprops

def create_rectangles_from_heatmap(heatmap):
    label_image = label(heatmap)
    rects = []
    
    for i, region in enumerate(regionprops(label_image)):
        bbox = region.bbox
        rect = ((bbox[1], bbox[0]), (bbox[3], bbox[2]))
        
        rects.append(rect)

    return rects


# In[4]:


import numpy as np
def create_mark(rectangles, shape): # [((begin_i, begin_j), (end_i, end_j)), ...]
    """
    Using a list of rectangles, create an image of marks
    """
    marked_img = np.zeros(shape = shape, dtype=int)
    for rect in rectangles:
        y = rect[1][1] - rect[0][1]
        marked_img[rect[0][1]:rect[1][1], rect[0][0]:rect[1][0], 3] = 255
    return marked_img


# In[5]:


import numpy as np
def apply_marks(origin, marks, inversed=True):
    """
    Extract numbers of the image using the marks
    """

    if inversed:
        condition = origin[:,:,0] < 240
    else:
        condition = origin[:,:,0] > 120

    return np.bitwise_and(marks[:,:,3] > 0, condition)


# In[6]:


import skimage.morphology
# le disk peut changer en fonction des vrai données
def fill_holes(img, elt = skimage.morphology.disk(1)):
    """
    Closure: to fill holes
    """
    if elt is None:
        return img
    return skimage.morphology.closing(img, elt)


# In[76]:


from skimage.measure import label, regionprops

def get_objects(img, ret_props=False):
    props = []
    objs = []
    label_image = label(img)
    
    for i, region in enumerate(regionprops(label_image)):
        bbox = region.bbox
        rect = ((bbox[1], bbox[0]), (bbox[3], bbox[2]))

        new_obj = np.zeros(shape=(bbox[2] - bbox[0], bbox[3] - bbox[1]), dtype=bool)
        new_obj[label_image[bbox[0]:bbox[2], bbox[1]:bbox[3]] == (i + 1)] = True
    
        props.append(region)
        objs.append(new_obj)

    if ret_props:
        return objs, props

    return objs


# In[73]:


import numpy as np
def get_objects_old(img):
    """
    From an image containing all the objects, return a list of image with a single object
    """
    tmp_img = np.copy(img) # to avoid modifying the original
    
    # list of objects
    objs = []
    
    def get_object(img, obj, i, j):
        # out of bounds
        if j >= img.shape[0] or j < 0:
            return
        if i >= img.shape[1] or i < 0:
            return
        # if objet
        if img[j][i]:
            obj[j][i] = True
            img[j][i] = False
            # recursive calls
            get_object(img, obj, i+1, j)
            get_object(img, obj, i, j+1)
            get_object(img, obj, i-1, j)
            get_object(img, obj, i, j-1)
    
    for j in range(tmp_img.shape[0]):
        for i in range(tmp_img.shape[1]):
            # first cell of object
            if tmp_img[j][i]:
                # create image for object
                obj = np.zeros(shape = tmp_img.shape, dtype=bool)
                # fill the image
                get_object(tmp_img, obj, i, j)
                # add to objects
                objs.append(obj)
                
    return objs


# In[8]:


import matplotlib.pyplot as plt
plt.rcParams['image.cmap'] = 'gray'
def show_images(objects, col=5):
    """
    Show a list of images (max 5 columns) = input
    """
    fig = plt.figure(figsize = (14,8))

    length = len(objects)
    
    rows = length // col + 1
    columns = col

    for i in range(length):
        fig.add_subplot(rows, columns, i + 1)
        plt.imshow(objects[i])
    plt.show()


# In[9]:


def get_bounding_box(img):
    min_i = img.shape[1]
    max_i = 0
    min_j = img.shape[0]
    max_j = 0
    
    for j in range(img.shape[0]):
        for i in range(img.shape[1]):
            if img[j][i]:
                if j > max_j:
                    max_j = j
                if j < min_j:
                    min_j = j
                if i > max_i:
                    max_i = i
                if i < min_i:
                    min_i = i
                    
    return ((min_i, min_j), (max_i + 1, max_j + 1))

import numpy as np
def transpose_obj_to_new_object(obj):
    """
    Create a smaller image to contain only to object
    """
    bbox = get_bounding_box(obj)
    shape = (bbox[1][1] - bbox[0][1], bbox[1][0] - bbox[0][0])
    # new image with size = bounding box
    new_obj = np.zeros(shape = shape, dtype=bool)
    
    # transpose
    for j in range(shape[0]):
        for i in range(shape[1]):
            new_obj[j][i] = obj[j + bbox[0][1]][i + bbox[0][0]]

    return new_obj

def transpose_objects(objects):
    """
    Transpose a list of objects
    """
    new_objects = []
    for obj in objects:
        # get object in a smaller image
        new_obj = transpose_obj_to_new_object(obj)
        new_objects.append(new_obj)
    return new_objects


# In[64]:


# filtrer: prendre uniquement les images avec un seul nombre (fonctionne uniquement si une majorité de nombres simples)
def extract_single_numbers(images):
    # compute mean of shapes
    shapes = [arr.shape for arr in images]
    mean_shape = np.mean(shapes, axis=0)

    # apply filter
    k = mean_shape / 2
    singles, multiples = [], []
    for obj in images:
        if (mean_shape[0] - k[0] <= obj.shape[0] and obj.shape[0] <= mean_shape[0] + k[0] and mean_shape[1] - k[1] <= obj.shape[1] and obj.shape[1] <= mean_shape[1] + k[1]):
            singles.append(obj)
        else:
            multiples.append(obj)
            
    return singles, multiples


# In[65]:


def turn_image(img): # turn if 3/4 * x > y
    if img.shape[0] < img.shape[1] * 0.75:
        return img[::-1].T # -90 degré # comment faire pour le sens ?
    return img

def turn_images(images):
    new_imgs = []
    for img in images:
        new_imgs.append(turn_image(img))
    return new_imgs


# In[75]:


# remove all black and almost all white objects
def filter_images(objects, props):
    objects_white = [pixels for pixels in objects if np.any(pixels)]
    real_objects = []
    real_props = []

    for i, obj in enumerate(objects_white):
        nb_zeros = np.count_nonzero(obj)
        nb_pixels = obj.shape[0] * obj.shape[1]
        if nb_zeros / nb_pixels < 0.65:
            real_objects.append(obj)
            real_props.append(props[i])

    return real_objects, real_props


# # With first image

# In[24]:


original = load_image("images/all-numbers.png")
show_image(original)


# In[25]:


rectangles = [
    (
        (3, 5), (42, 25)
    ),
    (
        (78, 5), (95, 25)
    ),
    (
        (3, 30), (30, 47)
    ),
    (
        (3, 77), (17, 95)
    ),
    (
        (15, 55), (42, 80)
    ),
    (
        (53, 55), (78, 73)
    )
]
marked = create_mark(rectangles, original.shape)
show_image(marked)


# In[26]:


extracted = apply_marks(original, marked)
show_image(extracted)


# In[27]:


closed = fill_holes(extracted)
show_image(closed)


# In[31]:


objects = get_objects(closed)
print(len(objects))
show_images(objects)


# # With another image

# In[33]:


img2 = load_image("images/190.png")
show_image(img2)


# In[34]:


rectangles = [
    (
        (10, 15), (60, 47)
    )
]
marked = create_mark(rectangles, img2.shape)
show_image(marked)


# In[35]:


t = apply_marks(img2, marked)
show_image(t)


# In[39]:


c = fill_holes(t, None)
show_image(c)


# In[40]:


objects = get_objects(c)
new_objects = transpose_objects(objects)
show_images(new_objects)


# # Main function

# In[78]:


def process(img, marked, elt = skimage.morphology.disk(1), inversed=True, ret_props=False):
    """
    From the original image, the marks and the structural element for closure,
    return and show the resulting objects in image
    """
    extracted = img if marked is None else apply_marks(img, marked, inversed)
    closed = fill_holes(extracted, elt)
    objects, props = get_objects(closed, True)
    #new_objects = transpose_objects(objects)
    real_objects, real_props = filter_images(objects, props)
    #singles, multiples = extract_single_numbers(real_objects)
    #turned = turn_images(singles) # not necessary
    turned = real_objects
    #turned.extend(multiples)
    show_images(turned[:10])

    if ret_props:
        return turned, real_props

    return turned


# # En 2 cells

# In[369]:


original = load_image("images/all-numbers.png")
rectangles = [
    (
        (3, 5), (42, 25)
    ),
    (
        (78, 5), (95, 25)
    ),
    (
        (3, 30), (30, 47)
    ),
    (
        (3, 77), (17, 95)
    ),
    (
        (15, 55), (42, 80)
    ),
    (
        (53, 55), (78, 73)
    )
]
marked = create_mark(rectangles, original.shape)
show_images([original, marked])


# In[370]:


n1 = process(original, marked, skimage.morphology.disk(1))


# In[371]:


img2 = load_image("images/overlap-and-holes.png")
rectangles = [
    (
        (20, 45), (140, 93)
    ),
    (
        (0, 88), (100, 200)
    ),
    (
        (108, 125), (185, 175)
    )
]
marked = create_mark(rectangles, img2.shape)
show_images([img2,marked])


# In[372]:


n2 = process(img2, marked, skimage.morphology.disk(3))


# # With traits

# In[373]:


img3 = load_image("images/trait_rotation.png")
rectangles = [
    (
        (10, 8), (120, 32)
    ),
    (
        (105, 40), (145, 80)
    ),
    (
        (45, 85), (100, 120)
    ),
    (
        (125, 128), (155, 170)
    ),
]
marked = create_mark(rectangles, img3.shape)
show_images([img3, marked])


# In[374]:


extracted = apply_marks(img3, marked)
show_image(extracted)


# In[375]:


n3 = process(img3, marked, skimage.morphology.disk(0))


# # With previous step output, ie my input

# In[376]:


input_img = load_image("images/input_1.png")
show_image(input_img)
print(input_img.shape)


# In[377]:


rectangles = [
    (
        (48, 245), (80, 300)
    ),
    (
        (135, 45), (170, 95)
    )
]
shape = (input_img.shape[0], input_img.shape[1], 4)
marked = create_mark(rectangles, shape)
show_images([input_img, marked], col=2)


# In[378]:


n4 = process(input_img, marked, skimage.morphology.disk(1), inversed=False)


# In[379]:


input_img2 = load_image("images/input_2.png")
show_image(input_img2)
print(input_img2.shape)


# In[385]:


rectangles = [
    (
        (5, 160), (60, 225)
    ),
    (
        (120, 15), (170, 85)
    ),
    (
        (230, 225), (270, 295)
    ),
    
    (
        (280, 170), (330, 240)
    ),
]
shape = (input_img2.shape[0], input_img2.shape[1], 4)
marked = create_mark(rectangles, shape)
show_images([input_img2, marked], col=2)


# In[386]:


img = input_img2
inversed=False
elt = skimage.morphology.disk(1)
extracted = apply_marks(img, marked, inversed)
closed = fill_holes(extracted, elt)
#show_image(extracted)
show_image(closed)


# In[389]:


n5 = process(input_img2, marked, skimage.morphology.disk(1), inversed=False)


# In[390]:


singles, multiples = extract_single_numbers(n5)
show_images(multiples)


# # TODO:
# 
# - Gérer les overlap

# In[442]:


import queue

def follow_path(original_squelette, pixel_coord):
    
    def next_cell(img, coords):
        x, y = coords
        img[y][x] = False
        cells = []
        for i in range(max(x - 1, 0), min(x + 2, img.shape[1])): # in the border of the image
            for j in range(max(y - 1, 0), min(y + 2, img.shape[0])):
                if img[j][i]:
                    cells.append((i, j))
        return cells
        
    new_sq = np.zeros(shape = original_squelette.shape, dtype=bool)
    
    next_cells = queue.Queue()
    next_cells.put(pixel_coord)
    
    while not next_cells.empty():
        cell = next_cells.get()
        new_sq[cell[1]][cell[0]] = True
        for c in next_cell(original_squelette, cell):
            next_cells.put(c)
    
    return new_sq


# In[437]:


# cut a squelette into 2 squelettes
def cut(original_sq, x, y):    
    # recup les pixels objet autour du point a enlever
    object_pixels = []
    
    for i in range(x - 1, x + 2):
        for j in range(y - 1, y + 2):
            if (i, j) == (x, y):
                continue
            
            if original_sq[j][i]:
                object_pixels.append((i, j))
            
    if len(object_pixels) != 2:
        return []
    
    original_sq[y][x] = False # set black
    
    objects = []
    for pix in object_pixels:
        objects.append(follow_path(original_sq, pix))
    
    return objects


# In[453]:


# recursif ?

def multiple_to_singles(original_img):
    # un overlap
    # calcul le squelette
    skelet = skimage.morphology.skeletonize(original_img)
    
    shape = obj.shape
    # a partir du centre, sur un rectangle de 1/3 de la taille de l'image:
    possible_cut = ((shape[0] * 2 // 5, shape[1] * 2 // 5), (shape[0] * 3 // 5, shape[1] * 3 // 5)) # ((y1, x1), (y2, x2))
    print(possible_cut)
    
    for j in range(possible_cut[0][0], possible_cut[1][0]):
        for i in range(possible_cut[0][1], possible_cut[1][1]):
            
            # si un pixel noir: suivant
            if not skelet[j][i]:
                continue
                
            # sinon: essaie de couper
            copy_sq = np.copy(skelet)
            new_objs = cut(copy_sq, i, j)
            if not new_objs:
                continue
            
            # + verifie l'ordre de grandeur sur le nombre de pixels blanc du squelette dans chaque
            obj1_nb_white = np.count_nonzero(new_objs[0])
            obj2_nb_white = np.count_nonzero(new_objs[1])
            div = obj1_nb_white / obj2_nb_white 
            if div > 2 or div < 1/2:
                continue
            print(div)
            
            print(i, j)
            show_image(skelet)
            return new_objs
    
    return []


# In[454]:


# get single object images and multiple object images
singles, multiples = extract_single_numbers(n5)

objects = []

# pour chaque image in multiples:
for obj in multiples:
    objects.extend(multiple_to_singles(obj))
            
show_images(objects)


# # Squelette
# 
# Comment je peux découper ces nombres ?

# In[393]:


singles, multiples = extract_single_numbers(n5)
skelet = skimage.morphology.skeletonize(multiples[0])
show_images([skelet, multiples[0]])


# In[397]:


# show squelettes
sq = []
for obj in n5:
    sq.append(skimage.morphology.skeletonize(obj))
show_images(sq)


# In[47]:


def get_proportions(new_objs):
    obj1_nb_white = np.count_nonzero(new_objs[0])
    obj2_nb_white = np.count_nonzero(new_objs[1])
    div = obj1_nb_white / obj2_nb_white 
    return div


def add_result(img_res, nb_pixels, results):
    """
    Add result only if good cut:
     - into 2 objects
     - good proportions
    """
    objects = get_objects(img_res)
    if len(objects) != 2: # if not 2 objects: bad cut
        return

    prop = get_proportions(objects)
    if prop <= 2 and prop >= 1/2: # good proportions
        res = {}
        # save the proportions
        res["prop"] = prop
        # save the number of cut pixels
        res["white_to_black"] = nb_pixels
        # save the results
        res["objects"] = objects

        results.append(res)


def pick_results(results):
    """
    Choose the best result:
     - less changed pixels (minimum cut)
     - best proportion if same cut
    """
    if len(results) == 0:
        print("No res to pick")
        return []

    min_cut = results[0]["white_to_black"]
    best_res = results[0]["objects"]
    best_prop = results[0]["prop"]

    for i in range(1, len(results)):
        if min_cut > results[i]["white_to_black"] or \
         (min_cut == results[i]["white_to_black"] and abs(best_prop - 1) > abs(results[i]["prop"] - 1)):
            min_cut = results[i]["white_to_black"]
            best_res = results[i]["objects"]

    return best_res


def multiple_to_single(original_img):
    # un overlap

    height, width = original_img.shape
    # cut the object with a line
    bounds_a = (-10, 10) # orientation of the line
    bounds_b = (0, 2 * width // 3) # x axis and y axis

    results = []

    # a negative
    for a in range(bounds_a[0], 0, 1):
        origin = 0

        for b in range(bounds_b[0], bounds_b[1]): # top bound
            # copy the image to not modify it
            tmp = np.copy(original_img)

            nb_pixels = 0

            # draw line <=> set white pixels to black
            for x_0 in range(0, width - b):
                y = origin - a * x_0
                x = x_0 + b

                y2 = origin - (a * (x_0 + 1))

                yend = min(y2, height)

                while y < yend:
                    if tmp[y][x]:
                        tmp[y][x] = 0
                        nb_pixels += 1
                    y += 1

            add_result(tmp, nb_pixels, results)

    # a positive
    for a in range(0, bounds_a[1], 1):
        origin = height - 1

        for b in range(bounds_b[0], bounds_b[1]): # bottom bound
            # copy the image to not modify it
            tmp = np.copy(original_img)

            nb_pixels = 0

            # draw line <=> set white pixels to black
            for x_0 in range(0, width):
                y = origin - a * x_0
                x = x_0 + b

                if x >= width:
                    break

                y2 = origin - a * (x_0 + 1)

                yend = max(y2, 0)

                while y > yend:
                    if not tmp[y][x]:
                        tmp[y][x] = 0
                        nb_pixels += 1
                    y -= 1

            add_result(tmp, nb_pixels, results)

    return pick_results(results)

def multiple_to_singles2(singles, original_imgs):
    """
    from list of single images and list of multiple images
    return all images as single number
    """
    tmp = []
    for img in original_imgs:
        res = multiple_to_single(img)
        if res:
            tmp.extend(res)
        else: # fail to cut
            tmp.append(img)

    # check ~ same shape
    height = 0
    width = 0
    for img in singles + tmp:
        height += img.shape[0]
        width += img.shape[1]
    length = len(singles) + len(tmp)
    height /= length
    width /= length

    while len(tmp) > 0:
        if tmp[0].shape[0] > 1.25 * height or tmp[0].shape[1] > 1.25 * width:
            res = multiple_to_single(tmp[0])
            if res:
                tmp.extend(res)
            else: # fail to cut
                singles.append(tmp[0])
        else:
            singles.append(tmp[0])
        tmp = tmp[1:]

    return singles


# In[48]:


s5, m5 = extract_single_numbers(n5)
show_images(m5)
show_images(multiple_to_singles2(s5, m5))


# In[52]:


s3, m3 = extract_single_numbers(n3)
show_images(m3)
singles = multiple_to_singles2(s3, m3)
show_images(singles)

# ## Pipeline

# In[92]:


import skimage.color
import json

def props_to_dict(props):
    return {
        'bbox': props.bbox,
        'minor_axis_length': props.minor_axis_length,
        'major_axis_length': props.major_axis_length,
        'centroid': props.centroid,
        'orientation': props.orientation,
    }

def process_from_heatmaps(inputFile, outputFile):
    heatmap = load_image(inputFile)
    
    rectangles = create_rectangles_from_heatmap(heatmap)

    images, props = process(heatmap, None, ret_props=True)
    singles, multiples = extract_single_numbers(images)
    res_images = multiple_to_singles2(singles, multiples)

    import os
    os.makedirs(outputFile, exist_ok=True)

    for i, image in enumerate(res_images):
        with open('os.path.join(outputFile, f"{i:04}.json")', 'w') as f:
            json.dump(props_to_dict(props[i]), f)
        skimage.io.imsave(os.path.join(outputFile, f"{i:04}.png"), image.astype(np.uint8)*255)


# In[93]:


process_from_heatmaps("output_dir/03_heatmaps.png", "results")


# In[ ]:




