import os
import math
import skimage.io
import cv2
from skimage.morphology import binary_dilation, dilation
from skimage.color import rgb2gray

from grid_detection import otsu_image

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
def extract_single_numbers(images, properties=None, digit_coeff=1e-1):
    # compute mean of shapes
    shapes = [arr.shape for arr in images]
    mean_shape = np.mean(shapes, axis=0)

    # apply filter
    k = mean_shape * digit_coeff
    singles, multiples = [], []
    singles_prop, multiples_prop = [], []
    for i, obj in enumerate(images):
        if (mean_shape[0] - k[0] <= obj.shape[0] and obj.shape[0] <= mean_shape[0] + k[0] and mean_shape[1] - k[1] <= obj.shape[1] and obj.shape[1] <= mean_shape[1] + k[1]):
            singles.append(obj)
            if properties:
                singles_prop.append(properties[i])
        else:
            multiples.append(obj)
            if properties:
                multiples_prop.append(properties[i])

    return singles, multiples, singles_prop, multiples_prop

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


class Properties:
    """
    Create properties by hand because we can to change some attributes
    """
    def __init__(self, label, merge_id, minor_axis_length, major_axis_length, orientation, corner_x, corner_y, bbox=None, centroid=None):
        self.label = label
        self.merge_id = merge_id
        self.bbox = bbox
        self.minor_axis_length = minor_axis_length
        self.major_axis_length = major_axis_length
        self.centroid = centroid
        self.orientation = orientation

        self.corner_x = corner_x
        self.corner_y = corner_y

    def update_bbox(self, old):
        self.bbox = (
            old[0] + self.corner_y,
            old[1] + self.corner_x,
            old[2] + self.corner_y,
            old[3] + self.corner_x
        )

    def update_centroid(self, old):
        self.centroid = (
            old[0] + self.corner_y,
            old[1] + self.corner_x
        )

def regionprop_to_properties(region_prop):
    """Convert skimage.RegionProp to Properties
    """
    if isinstance(region_prop, Properties):
        return region_prop

    return Properties(label=region_prop.label,
                      merge_id=region_prop.label,
                      minor_axis_length=region_prop.minor_axis_length,
                      major_axis_length=region_prop.major_axis_length,
                      orientation=region_prop.orientation,
                      corner_x=region_prop.bbox[0],
                      corner_y=region_prop.bbox[1],
                      bbox=region_prop.bbox,
                      centroid=region_prop.centroid
                      )


def add_result(img_res, nb_pixels, results, properties):
    """
    Add result only if good cut:
     - into 2 objects
     - good proportions
    """
    copy = np.copy(img_res)
    objects, tmp_properties = get_objects(img_res, ret_props=True)
    if len(objects) != 2: # if not 2 objects: bad cut
        return

    def get_proportions(new_objs):
        obj1_nb_white = np.count_nonzero(new_objs[0])
        obj2_nb_white = np.count_nonzero(new_objs[1])
        div = obj1_nb_white / obj2_nb_white
        return div

    prop = get_proportions(objects)
    if prop <= 2 and prop >= 1/2: # good proportions
        res = {}
        # save the proportions
        res["prop"] = prop
        # save the number of cut pixels
        res["white_to_black"] = nb_pixels
        # save the results
        res["objects"] = objects
        res["image"] = copy

        results.append((res, properties, tmp_properties))

def pick_results(results, nb_labels):
    """
    Choose the best result:
     - less changed pixels (minimum cut)
     - best proportion if same cut
    """
    if len(results) == 0:
        return [], None

    best_res = 0
    min_cut = results[0][0]["white_to_black"]
    best_prop = results[0][0]["prop"]

    for i in range(1, len(results)):
        if min_cut > results[i][0]["white_to_black"] or \
         (min_cut == results[i][0]["white_to_black"] and abs(best_prop - 1) > abs(results[i][0]["prop"] - 1)):
            min_cut = results[i][0]["white_to_black"]
            best_res = i

    # we use in postprocess: prop.bbox, minor_axis_length, major_axis_length, centroid, orientation

    res, old_prop, properties = results[best_res]

    top_y = old_prop.bbox[0]
    top_x = old_prop.bbox[1]

    merge_id = old_prop.label
    if isinstance(old_prop, Properties):
        merge_id = old_prop.merge_id

    # label of first object in cut = same as object before cut
    new_properties1 = Properties(
        old_prop.label,
        merge_id,
        properties[0].minor_axis_length,
        properties[0].major_axis_length,
        properties[0].orientation,
        top_x,
        top_y
    )
    # new label for the second object
    new_properties2 = Properties(
        nb_labels + 1,
        merge_id,
        properties[1].minor_axis_length,
        properties[1].major_axis_length,
        properties[1].orientation,
        top_x,
        top_y
    )

    # update the bbox to global image coords
    new_properties1.update_bbox(properties[0].bbox)
    new_properties2.update_bbox(properties[1].bbox)

    # update the bbox to global image coords
    new_properties1.update_centroid(properties[0].centroid)
    new_properties2.update_centroid(properties[1].centroid)

    return res["objects"], [new_properties1, new_properties2]


def cut_image(original_img, prop, nb_labels):
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

            add_result(tmp, nb_pixels, results, prop)

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

            add_result(tmp, nb_pixels, results, prop)

    return pick_results(results, nb_labels)

def multiples_to_singles(singles, original_imgs, props, m_props):
    """
    from list of single images and list of multiple images
    return all images as single number
    """
    # total nb images
    length = len(singles) + len(original_imgs)

    tmp = []
    tmp_props = []
    for i, img in enumerate(original_imgs):
        # try to cut
        res, res_props = cut_image(img, m_props[i], length)
        if res: # success
            tmp.extend(res)
            tmp_props.extend(res_props)
            length += 1
        else: # fail to cut
            tmp.append(img)
            tmp_props.append(m_props[i])

    # check ~ same shape
    height = 0
    width = 0
    for img in singles + tmp:
        height += img.shape[0]
        width += img.shape[1]
    height /= length
    width /= length

    while len(tmp) > 0:
        # if too large: multiple numbers
        if tmp[0].shape[0] > 1.5 * height or tmp[0].shape[1] > 1.5 * width:
            # try to cut
            res, res_props = cut_image(tmp[0], tmp_props[0], length)
            if res: # success
                tmp.extend(res)
                tmp_props.extend(res_props)
                length += 1
            else: # fail to cut
                singles.append(tmp[0])
                props.append(tmp_props[0])

        else:
            singles.append(tmp[0])
            props.append(tmp_props[0])

        # pop image from queue
        tmp = tmp[1:]
        tmp_props = tmp_props[1:]

    return singles, props


import skimage.color
import json

def props_to_dict(props, angle):
    return {
        'merge_id': props.merge_id,
        'bbox': props.bbox,
        'angle': angle,
        'minor_axis_length': props.minor_axis_length,
        'major_axis_length': props.major_axis_length,
        'centroid': props.centroid,
        'orientation': props.orientation,
    }

def apply_mask_binary(original, mask, prop):
    segment = rgb2gray(original[prop.bbox[0]:prop.bbox[2], prop.bbox[1]:prop.bbox[3]])
    bin_segment = binary_dilation(otsu_image(segment))

    bin_segment[~mask] = 0

    return bin_segment


def apply_mask_gray(original, prop):
    segment = (255 * original[prop.bbox[0]:prop.bbox[2], prop.bbox[1]:prop.bbox[3]]).astype(np.uint8)
    segment = dilation(255 - segment)
    denoised = cv2.fastNlMeansDenoising(segment)

    return denoised


def process_from_heatmaps(inputFile, heatmapFile, roadFile, outputFile):
    original = load_image(inputFile)
    heatmap = load_image(heatmapFile)
    roads = 255 - load_image(roadFile)

    os.makedirs(outputFile, exist_ok=True)

    images, props = process(heatmap, None, ret_props=True)
    # Binarize segments and apply mask from original image
    bin_images = [apply_mask_binary(original, mask, prop) for mask, prop in zip(images, props)]

    # cut multiple images to single one
    singles, multis, single_prop, mult_prop = extract_single_numbers(bin_images, props, digit_coeff=0.01)
    bin_images, props = multiples_to_singles(singles, multis, single_prop, mult_prop)
    props = [regionprop_to_properties(rp) for rp in props]

    for i, bin_im in enumerate(bin_images):
        skimage.io.imsave(os.path.join(outputFile, f"bin_{i:04}.png"), bin_im)

    original = rgb2gray(original)
    original_images = [apply_mask_gray(original, prop) for prop in props]

    for i, image in enumerate(original_images):
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
            json.dump(props_to_dict(props[i], 0), f)

        # Convert segment to grayscale and normalize for classification model
        skimage.io.imsave(os.path.join(outputFile, f"{i:04}.png"), image)


#process_from_heatmaps("output_dir/03_heatmaps.png", "results")
