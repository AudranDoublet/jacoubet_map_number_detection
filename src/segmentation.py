import math
import skimage.io
import cv2
from skimage.morphology import binary_dilation

def find_nearest_white(img, target):
    nonzero = cv2.findNonZero(img)

    if nonzero is None:
        return None

    distances = np.sqrt((nonzero[:,:,0] - target[0]) ** 2 + (nonzero[:,:,1] - target[1]) ** 2)
    nearest_index = np.argmin(distances)
    return nonzero[nearest_index][0]


def segment_orientation(segment):
    segment = binary_dilation(segment > 0) ^ (segment > 0)

    nonzero = cv2.findNonZero(segment * 255)

    if nonzero is None:
        return 0.0

    y_value = nonzero[:,:,0]

    x1 = nonzero[np.argmin(y_value)][0]
    x0 = nonzero[np.argmax(y_value)][0]

    v_dir = (x1[0] - x0[0], x1[1] - x0[1])
    angle = math.atan2(v_dir[1], v_dir[0])

    return math.degrees(angle)


def load_image(image_path):
    """
    Load image with its path
    """
    return skimage.io.imread(image_path)


import matplotlib.pyplot as plt
plt.rcParams['image.cmap'] = 'gray'
def show_image(img):
    """
    Show a single image
    """
    fig = plt.figure(figsize = (15,10))
    plt.imshow(img)


from skimage.measure import label, regionprops

def create_rectangles_from_heatmap(heatmap):
    label_image = label(heatmap)
    rects = []

    for i, region in enumerate(regionprops(label_image)):
        bbox = region.bbox
        rect = ((bbox[1], bbox[0]), (bbox[3], bbox[2]))

        rects.append(rect)

    return rects


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


import skimage.morphology
# le disk peut changer en fonction des vrai données
def fill_holes(img, elt = skimage.morphology.disk(1)):
    """
    Closure: to fill holes
    """
    if elt is None:
        return img
    return skimage.morphology.closing(img, elt)


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


def turn_image(img): # turn if 3/4 * x > y
    if img.shape[0] < img.shape[1] * 0.75:
        return img[::-1].T # -90 degré # comment faire pour le sens ?
    return img

def turn_images(images):
    new_imgs = []
    for img in images:
        new_imgs.append(turn_image(img))
    return new_imgs


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

def process(img, marked, elt = skimage.morphology.disk(1), inversed=True, ret_props=False):
    """
    From the original image, the marks and the structural element for closure,
    return and show the resulting objects in image
    """
    extracted = img if marked is None else apply_marks(img, marked, inversed)
    closed = fill_holes(extracted, elt)
    objects, props = get_objects(closed, True)

    real_objects, real_props = filter_images(objects, props)
    turned = real_objects

    if ret_props:
        return turned, real_props

    return turned


import skimage.color
import json

def props_to_dict(props, angle):
    return {
        'bbox': props.bbox,
        'angle': angle,
        'minor_axis_length': props.minor_axis_length,
        'major_axis_length': props.major_axis_length,
        'centroid': props.centroid,
        'orientation': props.orientation,
    }

def process_from_heatmaps(inputFile, roadFile, outputFile):
    heatmap = load_image(inputFile)
    roads = 255 - load_image(roadFile)

    rectangles = create_rectangles_from_heatmap(heatmap)

    images, props = process(heatmap, None, ret_props=True)

    import os
    os.makedirs(outputFile, exist_ok=True)

    for i, image in enumerate(images):
        pos = props[i].centroid
        pos = (int(pos[0]), int(pos[1]))

        nearest_road = find_nearest_white(roads[pos[0]-30:pos[0]+30, pos[1]-30:pos[1]+30], [30, 30])

        if nearest_road is None:
            continue

        nearest_road = [nearest_road[1], nearest_road[0]]

        nearest_road[0] += pos[0] - 30
        nearest_road[1] += pos[1] - 30

        angle = segment_orientation(roads[ nearest_road[0]-30:nearest_road[0]+30, nearest_road[1]-30:nearest_road[1]+30 ])

        image = skimage.transform.rotate(image * 1.0, -angle, resize=True)


        with open(os.path.join(outputFile, f"{i:04}.json"), 'w') as f:
            json.dump(props_to_dict(props[i], angle), f)
        skimage.io.imsave(os.path.join(outputFile, f"{i:04}.png"), (image * 255).astype(np.uint8))


#process_from_heatmaps("output_dir/03_heatmaps.png", "results")
