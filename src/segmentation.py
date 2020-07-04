import os
import math
import skimage.io
import cv2
from skimage.morphology import binary_dilation, dilation
from skimage.color import rgb2gray
from skimage import img_as_ubyte

from grid_detection import otsu_image

K_SEUIL_MULTIPLE = 0.62
K_SEUIL_MULTIPLE2 = 1.7
INDEX = 0


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


def remove_noise(images, properties=None):
    shapes = [arr.shape for arr in images]
    mean_shape = np.mean(shapes, axis=0)
    k2 = mean_shape * 0.4
    seuil = 33 # pixels

    suppr = []
    results = []
    results_prop = []

    for i, obj in enumerate(images):
        if (obj.shape[0] <= k2[0] or obj.shape[1] <= k2[1] or obj.shape[1] + obj.shape[0] < seuil):
            suppr.append(obj)
        else:
            results.append(obj)
            results_prop.append(properties[i])

    return results, results_prop, suppr

# filtrer: prendre uniquement les images avec un seul nombre (fonctionne uniquement si une majorité de nombres simples)
def extract_single_numbers(images, properties=None):
    # compute mean of shapes
    shapes = [arr.shape for arr in images]
    mean_shape = np.mean(shapes, axis=0)

    # apply filter
    k = mean_shape * K_SEUIL_MULTIPLE
    singles, multiples = [], []
    suppr = []
    singles_prop, multiples_prop = [], []
    for i, obj in enumerate(images):
        if (obj.shape[0] <= k[0] or obj.shape[1] <= k[1]):
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
    if len(objects) < 2: # if 1 or 0 object: bad cut
        return

    def get_proportions(new_objs):
        obj1_nb_white = np.count_nonzero(new_objs[0])
        obj2_nb_white = np.count_nonzero(new_objs[1])
        div = obj1_nb_white / obj2_nb_white
        return div

    prop = get_proportions(objects)

    if prop <= 5 and prop >= 1/5: # good proportions
        res = {}
        # save the proportions
        res["prop"] = prop
        # save the number of cut pixels
        res["white_to_black"] = nb_pixels
        # save the results
        res["objects"] = objects
        res["image"] = copy
        res["nb_objects"] = len(objects)

        results.append((res, properties, tmp_properties))

def pick_results(results, nb_labels):
    """
    Choose the best result:
     - less changed pixels (minimum cut)
     - best proportion if same cut
    """
    if len(results) == 0:
        return [], None, nb_labels

    best_res = 0
    min_cut = results[0][0]["white_to_black"]
    best_prop = results[0][0]["prop"]
    best_nb = results[0][0]["nb_objects"]

    for i in range(1, len(results)):
        if best_nb > results[i][0]["nb_objects"] or \
         min_cut > results[i][0]["white_to_black"] or \
         (min_cut == results[i][0]["white_to_black"] and abs(best_prop - 1) > abs(results[i][0]["prop"] - 1)):
            min_cut = results[i][0]["white_to_black"]
            best_res = i
            best_nb = results[i][0]["nb_objects"]

    # we use in postprocess: prop.bbox, minor_axis_length, major_axis_length, centroid, orientation

    res, old_prop, properties = results[best_res]

    top_y = old_prop.bbox[0]
    top_x = old_prop.bbox[1]

    merge_id = old_prop.label
    if isinstance(old_prop, Properties):
        merge_id = old_prop.merge_id

    # label of first object in cut = same as object before cut
    new_properties = []
    new_properties.append(
        Properties(
            old_prop.label,
            merge_id,
            properties[0].minor_axis_length,
            properties[0].major_axis_length,
            properties[0].orientation,
            top_x,
            top_y
        )
    )
    # new label for the second object
    for i in range(1, best_nb):
        nb_labels += 1
        new_properties.append(
            Properties(
                merge_id,
                nb_labels,
                properties[i].minor_axis_length,
                properties[i].major_axis_length,
                properties[i].orientation,
                top_x,
                top_y
            )
        )

    # update the bbox to global image coords
    for i in range(best_nb):
        new_properties[i].update_bbox(properties[i].bbox)

    # update the bbox to global image coords
    for i in range(best_nb):
        new_properties[i].update_centroid(properties[i].centroid)

    return res["objects"], new_properties, nb_labels


def cut_image(original_img, prop, nb_labels):
    # un overlap

    height, width = original_img.shape
    # cut the object with a line
    bounds_a = (-10, 10) # orientation of the line
    bounds_b = (0, 3 * width // 4) # x axis and y axis

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


def cut_image_with_contours(image, old_prop, padding=5):
    image = image.astype(np.uint8) * 255

    cnts = cv2.findContours(image, cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0]

    if len(cnts) < 2:
        return [image], [old_prop]

    top_y = old_prop.bbox[0]
    top_x = old_prop.bbox[1]

    merge_id = old_prop.label
    if isinstance(old_prop, Properties):
        merge_id = old_prop.merge_id

    new_segments = []
    new_props = []
    for cnt in cnts:
        # If object is too small, discard it
        if cv2.contourArea(cnt) < 30:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        # Create new segment
        new_segment = image[y:y + h, x:x + h].copy()
        new_prop = Properties(
            old_prop.label + 1,
            merge_id,
            int(w),
            int(h),
            0,
            top_x,
            top_y
        )

        # update the bbox to global image coords
        new_prop.update_bbox((int(x - padding), int(y - padding), int(x + w + padding), int(y + h + padding)))

        # update the bbox to global image coords
        new_prop.update_centroid((int((x + x + w) // 2), int((y + y + h) // 2)))

        new_segments.append(new_segment)
        new_props.append(new_prop)

    return new_segments, new_props


def multiples_to_singles(singles, original_imgs, props, m_props, outputFile):
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
        res, res_props, length = cut_image(img, m_props[i], length)
        if res: # success
            #skimage.io.imsave(os.path.join(outputFile, f"cutting_{i:04}.png"), img_as_ubyte(img, True))
            #for i2, res_i in enumerate(res):
            #    skimage.io.imsave(os.path.join(outputFile, f"cutting_{i:04}_into{i2}.png"), img_as_ubyte(res_i, True))
            tmp.extend(res)
            tmp_props.extend(res_props)
        else: # fail to cut
            skimage.io.imsave(f"fail_cut_{i:04}.png", img_as_ubyte(img, True))
            tmp.append(img)
            tmp_props.append(m_props[i])

    # check ~ same shape
    shapes = [arr.shape for arr in singles + tmp]
    mean_shape = np.mean(shapes, axis=0)

    k = 0

    singles.extend(tmp)
    props.extend(tmp_props)
    """
    while len(tmp) > 0:
        # if too large: multiple numbers
        if tmp[0].shape[1] > K_SEUIL_MULTIPLE2 * mean_shape[1]:
            # try to cut
            res, res_props, length = cut_image(tmp[0], tmp_props[0], length)
            if res: # success
                skimage.io.imsave(f"cutting2_{k:04}.png", img_as_ubyte(tmp[0], True))
                for i2, res_i in enumerate(res):
                    skimage.io.imsave(f"cutting2_{k:04}_into{i2}.png", img_as_ubyte(res_i, True))
                k += 1
                tmp.extend(res)
                tmp_props.extend(res_props)
            else: # fail to cut
                res, res_props = cut_image_with_contours(tmp[0], tmp_props[0])
                singles.extend(res)
                props.extend(res_props)

        else:
            singles.append(tmp[0])
            props.append(tmp_props[0])

        # pop image from queue
        tmp = tmp[1:]
        tmp_props = tmp_props[1:]
    """
    return singles, props

def remove_end_noise(images, properties=None):
    shapes = [arr.shape for arr in images]
    mean_shape = np.mean(shapes, axis=0)
    k2 = mean_shape * 0.1
    seuil = 25 # pixels

    suppr = []
    results = []
    results_prop = []

    for i, obj in enumerate(images):
        if (obj.shape[0] <= k2[0] or obj.shape[1] <= k2[1] or obj.shape[1] + obj.shape[0] < seuil):
            suppr.append(obj)
        else:
            results.append(obj)
            results_prop.append(properties[i])

    return results, results_prop, suppr


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
    roads = load_image(roadFile)

    os.makedirs(outputFile, exist_ok=True)

    images, props = process(heatmap, None, ret_props=True)
    # Binarize segments and apply mask from original image
    bin_images = [apply_mask_binary(original, mask, prop) for mask, prop in zip(images, props)]

    bin_images, props2, suppr = remove_noise(bin_images, props)
    # cut multiple images to single one
    singles, multis, single_prop, mult_prop = extract_single_numbers(bin_images, props2)
    # debug
    for i in range(len(suppr)):
        skimage.io.imsave(os.path.join(outputFile, f"suppr_{i:04}.png"), img_as_ubyte(suppr[i], True))
    for i in range(len(singles)):
        skimage.io.imsave(os.path.join(outputFile, f"single_{i:04}.png"), img_as_ubyte(singles[i], True))
    for i in range(len(multis)):
        skimage.io.imsave(os.path.join(outputFile, f"multi_{i:04}.png"), img_as_ubyte(multis[i], True))

    bin_images, props = multiples_to_singles(singles, multis, single_prop, mult_prop, outputFile)

    bin_images_end, props_end, suppr_end = remove_end_noise(bin_images, props)
    # debug
    #for i in range(len(suppr_end)):
    #    skimage.io.imsave(os.path.join(outputFile, f"suppr-end_{i:04}.png"), img_as_ubyte(suppr_end[i], True))

    props = [regionprop_to_properties(rp) for rp in props_end]

    for i, bin_im in enumerate(bin_images_end):
        #skimage.io.imsave(os.path.join(outputFile, f"bin_{i:04}.png"), (255 * bin_im).astype(np.uint8))
        skimage.io.imsave(os.path.join(outputFile, f"bin_{i:04}.png"), img_as_ubyte(bin_im, True))

    original = rgb2gray(original)
    original_images = [apply_mask_gray(original, prop) for prop in props]

    size = 40

    for i, image in enumerate(original_images):
        pos = props[i].centroid
        pos = (int(pos[0]), int(pos[1]))

        nearest_road = find_nearest_white(roads[pos[0]-size:pos[0]+size, pos[1]-size:pos[1]+size], [size, size])

        if nearest_road is not None:
            nearest_road = [nearest_road[1], nearest_road[0]]

            nearest_road[0] += pos[0] - size
            nearest_road[1] += pos[1] - size

            angle = segment_orientation(roads[ nearest_road[0]-size:nearest_road[0]+size, nearest_road[1]-size:nearest_road[1]+size ]) % 90
        else:
            angle = 0

        rotate = skimage.transform.rotate(image * 1.0, -angle, resize=True)

        with open(os.path.join(outputFile, f"{i:04}.json"), 'w') as f:
            json.dump(props_to_dict(props[i], 0), f)

        # Convert segment to grayscale and normalize for classification model
        skimage.io.imsave(os.path.join(outputFile, f"{i:04}_unrotate.png"), image.astype(np.uint8))
        skimage.io.imsave(os.path.join(outputFile, f"{i:04}.png"), rotate.astype(np.uint8))


#process_from_heatmaps("output_dir/03_heatmaps.png", "results")
