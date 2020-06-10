import csv
import os
import json
import sys

import cv2
import skimage.io
import numpy as np


def rewarp_coord(coords, rewarp_matrix):
    if rewarp_matrix is None:
        return coords

    # cv2 reads coordinates as x, y
    coords = np.array([[coords[1], coords[0]]], dtype=np.float64)

    return cv2.perspectiveTransform(coords[:, np.newaxis], rewarp_matrix)[0, 0]


def get_detection_grid(image_path, detections, dewarp_matrix):
    print(image_path)
    image = skimage.io.imread(image_path)
    height, width = image.shape[:2]

    height = int(np.ceil(height // 100))
    width = int(np.ceil(width // 100))

    rewarp_matrix = np.linalg.pinv(dewarp_matrix) if dewarp_matrix is not None else None

    grid = [[0] * width for i in range(height)]
    for i in range(height):
        for j in range(width):
            grid[i][j] = []

    for detect in detections:
        orig_coords = rewarp_coord(detect['coords'], rewarp_matrix)
        i = int(orig_coords[1]) // 100
        j = int(orig_coords[0]) // 100

        grid[i][j].append(int(detect['label']))

    return grid, height, width


def dump_submission(image_path, detections, dewarp_matrix_file, output_file):
    dewarp_matrix = np.loadtxt(dewarp_matrix_file, delimiter=',')

    grid, height, width = get_detection_grid(image_path, detections, dewarp_matrix)

    atlas_id = os.path.splitext(os.path.basename(image_path))[0]
    with open(output_file, 'a') as file:
        csv_writer = csv.writer(file)
        for i in range(height):
            for j in range(width):
                tile_id = f'{atlas_id}_{j}_{i}'
                if len(grid[i][j]):
                    labels = ' '.join(str(label) for label in sorted(grid[i][j]))
                else:
                    labels = ''

                csv_writer.writerows([[tile_id] + [labels]])


def process_file(input_file, detection_file, dewarp_matrix_file, output_file):
    print(f'Postprocess {input_file}')
    with open(detection_file) as f:
        detections = json.load(f)

        dump_submission(input_file, detections, dewarp_matrix_file, output_file)


if __name__ == '__main__':

    assert len(sys.argv) == 4, "bad arguments"

    _, input_file, detection_file, dewarp_matrix_file, output_file = sys.argv

    process_file(input_file, detection_file, dewarp_matrix_file, output_file)


