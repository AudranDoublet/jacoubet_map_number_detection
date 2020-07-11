import os
import math
import skimage.io
import cv2
import numpy as np
from skimage.morphology import *
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu
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


def get_nearest_road(pos, roads, search_radius):
    mn = np.array(pos) - search_radius
    mx = np.array(pos) + search_radius

    road = roads[mn[0]:mx[0], mn[1]:mx[1]]
    nearest_road = find_nearest_white(road, [search_radius, search_radius])

    if nearest_road is None:
        print("euh")
        return None

    nearest_road = np.array([nearest_road[1], nearest_road[0]])

    return nearest_road + mn


def segment_orientation(segment):
    segment = binary_dilation(segment > 0) ^ (segment > 0)

    nonzero = cv2.findNonZero(segment * 255)

    if nonzero is None:
        return 0.0

    y_value = nonzero[:,:,0]

    x1 = nonzero[np.argmin(y_value)][0]
    x0 = nonzero[np.argmax(y_value)][0]

    v_dir = (x1[0] - x0[0], x0[1] - x1[1])
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


import skimage.morphology
def fill_holes(img, elt = skimage.morphology.disk(1)):
    """
    Closure: to fill holes
    """
    if elt is None:
        return img
    return skimage.morphology.closing(img, elt)


from skimage.measure import label, regionprops
def get_objects(img, ret_props=False):
    """
    Find the connected components and get their properties in the image
    """
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


def count_lines(binary, threshold1=0.3, threshold2=0.5, threshold2_width=2):
    count = 0

    for reg in regionprops(label(binary)):
        if reg.major_axis_length == 0.0:
            continue

        ratio = reg.minor_axis_length / reg.major_axis_length

        if ratio < threshold1 or (ratio < threshold2 and reg.minor_axis_length <= threshold2_width):
            count += 1

    return count


def check_has_small_holes(image):
    h, w = image.shape
    cls = 10

    result = np.zeros((h + cls*2, w + cls*2))
    result[cls:-cls,cls:-cls] = image

    binary = result != 0
    filtered = result != 0

    filtered = remove_small_holes(filtered, 10)
    return np.sum(filtered ^ binary) > 10


def check_is_striped_zone(binary):
    line_count = count_lines(np.bitwise_not(binary)) + count_lines(binary)

    bg_count = np.sum(1 - binary)

    if line_count > 3:
        return True

    perimeter = sum([reg.perimeter for reg in regionprops(label(binary))])
    min_lines = 2 if perimeter > 100 else 3

    # create a larger image to avoid filling holes/close due to border
    if count_lines(np.bitwise_not(binary)) < min_lines:
        h, w = binary.shape
        cls = 10

        result = np.zeros((h + cls*2, w + cls*2))
        result[cls:-cls,cls:-cls] = binary

        binary = result
        binary = binary != 0

    # check if we are in a striped zone
    sm1 = np.sum(binary_closing(binary, square(3)) ^ binary)
    sm2 = np.sum(remove_small_holes(binary, 100) ^ binary)

    def test(v, threshold=80, perc=0.3):
        return v > threshold or v / np.sum(binary) > perc

    return (test(sm1) and test(sm2)) or test(sm1, min(150, bg_count / 2), 0.5) or test(sm2, 160, 0.5)


def check_if_empty(binary):
    empty = True

    for reg in regionprops(label(binary)):
        if reg.area / reg.filled_area > 0.6:
            empty = False

    return empty


def remove_lines_and_circle(binary, threshold1=0.2, threshold_poly=0.9):
    count = 0
    lbl = label(binary, connectivity=1)

    result = binary.copy()

    for reg in regionprops(lbl):
        if reg.major_axis_length == 0.0:
            continue

        ratio = reg.minor_axis_length / reg.major_axis_length
        ratio_area = reg.filled_area / reg.convex_area
        area = reg.area / reg.filled_area

        if ratio_area > 1.0 or area < 0.9:
            ratio_area = 0.0

        if ratio < threshold1 or (ratio > 0.9 and area > 0.9) or (ratio_area > threshold_poly and ratio > 0.4):
            result[lbl == reg.label] = 0


    return result


def check_not_geometric(image):
    h, w = image.shape
    cls = 10

    result = np.zeros((h + cls*2, w + cls*2))
    result[cls:-cls,cls:-cls] = image

    binary = result != 0
    filtered = result != 0

    filtered = remove_lines_and_circle(binary)
    filtered = remove_small_objects(filtered, 30)

    return np.sum(filtered) < 20


def tclosing(image):
    h, w = image.shape
    cls = 10

    result = np.zeros((h + cls*2, w + cls*2))
    result[cls:-cls,cls:-cls] = image

    return binary_closing(result != 0, square(cls))


def check_is_line(image):
    closing = tclosing(image)

    for reg in regionprops(label(closing)):
        if reg.minor_axis_length / reg.major_axis_length < 0.2:
            return True

    return False


def check_is_thick(binary):
    for reg in regionprops(label(binary)):
        perc = reg.area / reg.perimeter

        if perc > 2.2:
            return True

    return False


def remove_noise(images, properties=None):
    shapes = [arr.shape for arr in images]
    mean_shape = np.mean(shapes, axis=0)
    k2 = mean_shape * 0.4
    seuil = 33 # pixels

    suppr = []
    results = []
    results_prop = []

    for i, obj in enumerate(images):
        results.append(obj)
        results_prop.append(properties[i])

    return results, results_prop, suppr


def extract_single_numbers(images, properties=None):
    # compute mean of shapes
    shapes = [arr.shape for arr in images]
    mean_shape = np.mean(shapes, axis=0)

    # apply filter
    k = mean_shape * K_SEUIL_MULTIPLE
    singles, multiples = [], []
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


def process(img):
    """
    From the original image
    return the resulting objects in image
    """
    closed = fill_holes(img)
    objects, props = get_objects(closed, True)

    return objects, props


class Properties:
    """
    Create properties by hand because we want to change some attributes
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
     - into 2 or more objects
     - good proportions
    """
    copy = np.copy(img_res)
    objects, tmp_properties = get_objects(img_res, ret_props=True)
    if len(objects) < 2: # if 1 or 0 object: bad cut
        return

    def get_proportions(new_objs):
        nb_white = [np.count_nonzero(obj) for obj in new_objs]
        nbw = sorted(nb_white, reverse=True)
        prop = nbw[0] / nbw[1]
        return prop

    prop = get_proportions(objects)

    k = 2.5
    if prop > k or prop < 1/k: # bad proportion
        return

    res = {}
    # save the proportions
    res["prop"] = prop
    # save the number of cut pixels
    res["white_to_black"] = nb_pixels
    # save the results
    res["objects"] = objects
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
                nb_labels,
                merge_id,
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
            tmp.extend(res)
            tmp_props.extend(res_props)
        else: # fail to cut
            skimage.io.imsave(os.path.join(outputFile, f"fail_cut_{i:04}.png"), img_as_ubyte(img, True))
            tmp.append(img)
            tmp_props.append(m_props[i])

    singles.extend(tmp)
    props.extend(tmp_props)
    return singles, props


def remove_end_noise(images, properties=None):
    """
    Remove images that are too small
    """
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


from sklearn.neighbors import NearestNeighbors
from skimage.metrics import peak_signal_noise_ratio

def recreate_image_with_shape(image, shape):
    result = np.zeros(shape, dtype=image.dtype)

    dy = (shape[0] - image.shape[0]) // 2
    dx = (shape[1] - image.shape[1]) // 2

    result[dy:dy+image.shape[0], dx:dx+image.shape[1]] = image
    return result


def postprocess_filter_distance(outputFile, props, original_images):
    current = 0

    out_prop = []
    out_img  = []

    centroids = np.array([prop.centroid for prop in props])

    knn = NearestNeighbors(n_neighbors=4).fit(centroids)
    distances, _ = knn.kneighbors(centroids)

    for distances, p, img in zip(distances, props, original_images):
        if distances[1] > 500:
            skimage.io.imsave(os.path.join(outputFile, f"postprocess_suppr_far_away_{current:04}.png"), img_as_ubyte(img, True))
            current += 1

        else:
            out_prop.append(p)
            out_img.append(img)

    return out_prop, out_img


def postprocess_filter_similar(outputFile, props, original_images):
    current = 0

    out_prop = []
    out_img  = []

    centroids = np.array([prop.centroid for prop in props])

    knn = NearestNeighbors(n_neighbors=4).fit(centroids)
    distances, indices = knn.kneighbors(centroids)

    for distances, indices, p, img in zip(distances, indices, props, original_images):
        self_idx = indices[0]
        psnr = 1e9

        for dist, idx in zip(distances[1:], indices[1:]):
            if dist > 500:
                break

            a = original_images[self_idx]
            b = original_images[idx]

            shape = (max(a.shape[0], b.shape[0]), max(a.shape[1], b.shape[1]))

            a = recreate_image_with_shape(a, shape)
            b = recreate_image_with_shape(b, shape)

            psnr = min(peak_signal_noise_ratio(a, b), psnr)

        if psnr > 9 and psnr < 1e9:
            skimage.io.imsave(os.path.join(outputFile, f"postprocess_suppr_neighbors_too_similar_{current:04}.png"), img_as_ubyte(img, True))
            current += 1

        else:
            out_prop.append(p)
            out_img.append(img)

    return out_prop, out_img


def postprocess_filter(outputFile, props, original_images):
    current = 0

    out_prop = []
    out_img  = []

    for p, img in zip(props, original_images):
        binary = img > threshold_otsu(img)

        if check_is_striped_zone(binary):
            skimage.io.imsave(os.path.join(outputFile, f"postprocess_suppr_striped_{current:04}.png"), img_as_ubyte(img, True))
            current += 1

        elif threshold_otsu(img) < 110:
            skimage.io.imsave(os.path.join(outputFile, f"postprocess_suppr_too_dark_{current:04}.png"), img_as_ubyte(img, True))
            current += 1

        elif check_if_empty(binary):
            skimage.io.imsave(os.path.join(outputFile, f"postprocess_suppr_empty_{current:04}.png"), img_as_ubyte(img, True))
            current += 1

        elif check_not_geometric(binary):
            skimage.io.imsave(os.path.join(outputFile, f"postprocess_suppr_geometric_{current:04}.png"), img_as_ubyte(img, True))
            current += 1

        elif check_is_line(binary):
            skimage.io.imsave(os.path.join(outputFile, f"postprocess_suppr_line_{current:04}.png"), img_as_ubyte(img, True))
            current += 1

        elif check_is_thick(binary):
            skimage.io.imsave(os.path.join(outputFile, f"postprocess_suppr_thick_{current:04}.png"), img_as_ubyte(img, True))
            current += 1

        elif check_has_small_holes(binary):
            skimage.io.imsave(os.path.join(outputFile, f"postprocess_suppr_small_holes_{current:04}.png"), img_as_ubyte(img, True))
            current += 1

        elif binary.shape[0] + binary.shape[1] > 64:
            skimage.io.imsave(os.path.join(outputFile, f"postprocess_suppr_too_big_{current:04}.png"), img_as_ubyte(img, True))
            current += 1

        else:
            out_prop.append(p)
            out_img.append(img)

    out_prop, out_img = postprocess_filter_similar(outputFile, out_prop, out_img)
    out_prop, out_img = postprocess_filter_distance(outputFile, out_prop, out_img)

    return out_prop, out_img


def get_original_masked(original, heatmap):
    segment = rgb2gray(original)
    segment = (255 * segment).astype(np.uint8)
    for i in range(segment.shape[0]):
        for j in range(segment.shape[1]):
            if heatmap[i][j] == 0:
                segment[i][j] = 0
    return segment


from PIL import ImageFont, ImageDraw, Image
def save_processed_heatmap(segment, props, outputFile, idx):
    """
    Save heatmap image with rectangles as detected numbers
    """
    img = Image.fromarray(segment)
    draw = ImageDraw.Draw(img)

    for prop in props:
        bbox = prop.bbox
        box = [bbox[1], bbox[0], bbox[3], bbox[2]]
        draw.rectangle(box, outline=150)

    img.save(os.path.join(outputFile, f"heatmaps_objects_{idx}.png"))


def process_from_heatmaps(inputFile, heatmapFile, roadFile, outputFile):
    DEBUG = False
    original = load_image(inputFile)
    heatmap = load_image(heatmapFile)
    roads = load_image(roadFile)
    if DEBUG:
        masked = get_original_masked(original, heatmap)

    os.makedirs(outputFile, exist_ok=True)

    images, props = process(heatmap)
    if DEBUG:
        save_processed_heatmap(masked, props, outputFile, 1)

    # Binarize segments and apply mask from original image
    bin_images = [apply_mask_binary(original, mask, prop) for mask, prop in zip(images, props)]
    if DEBUG:
        for i in range(len(bin_images)):
            skimage.io.imsave(os.path.join(outputFile, f"bin1_{i:04}.png"), img_as_ubyte(bin_images[i], True))

    bin_images, props2, suppr = remove_noise(bin_images, props)
    if DEBUG:
        save_processed_heatmap(masked, props2, outputFile, 2)

    # cut multiple images to single one
    singles, multis, single_prop, mult_prop = extract_single_numbers(bin_images, props2)
    if DEBUG:
        save_processed_heatmap(masked, single_prop + mult_prop, outputFile, 3)

        for i in range(len(suppr)):
            skimage.io.imsave(os.path.join(outputFile, f"suppr_{i:04}.png"), img_as_ubyte(suppr[i], True))
        for i in range(len(singles)):
            skimage.io.imsave(os.path.join(outputFile, f"single_{i:04}.png"), img_as_ubyte(singles[i], True))
        for i in range(len(multis)):
            skimage.io.imsave(os.path.join(outputFile, f"multi_{i:04}.png"), img_as_ubyte(multis[i], True))

    bin_images, props = multiples_to_singles(singles, multis, single_prop, mult_prop, outputFile)
    if DEBUG:
        save_processed_heatmap(masked, props, outputFile, 4)

    bin_images_end, props_end, suppr_end = remove_end_noise(bin_images, props)
    if DEBUG:
        save_processed_heatmap(masked, props_end, outputFile, 5)

        for i in range(len(suppr_end)):
            skimage.io.imsave(os.path.join(outputFile, f"suppr-end_{i:04}.png"), img_as_ubyte(suppr_end[i], True))

    props = [regionprop_to_properties(rp) for rp in props_end]

    for i, bin_im in enumerate(bin_images_end):
        skimage.io.imsave(os.path.join(outputFile, f"bin_{i:04}.png"), img_as_ubyte(bin_im, True))

    original = rgb2gray(original)
    original_images = [apply_mask_gray(original, prop) for prop in props]

    props, original_images = postprocess_filter(outputFile, props, original_images)
    if DEBUG:
        save_processed_heatmap(masked, props, outputFile, 6)

    size = 40

    for i, image in enumerate(original_images):
        pos = props[i].centroid
        pos = (int(pos[0]), int(pos[1]))

        nearest_road = get_nearest_road(pos, roads, 100)

        if nearest_road is not None:
            mn = nearest_road - size
            mx = nearest_road + size

            sample = roads[mn[0]:mx[0], mn[1]:mx[1]]
            sample = remove_small_holes(sample != 0, 512) * 1

            angle = segment_orientation(sample)

            if angle < -90:
                angle += 180

            if angle > 100:
                angle -= 180

        else:
            angle = 0

        rotate = skimage.transform.rotate(image * 1.0, -angle, resize=True)

        with open(os.path.join(outputFile, f"{i:04}.json"), 'w') as f:
            json.dump(props_to_dict(props[i], angle), f)

        # Convert segment to grayscale and normalize for classification model
        skimage.io.imsave(os.path.join(outputFile, f"{i:04}_unrotate.png"), image.astype(np.uint8))
        skimage.io.imsave(os.path.join(outputFile, f"{i:04}.png"), rotate.astype(np.uint8))
