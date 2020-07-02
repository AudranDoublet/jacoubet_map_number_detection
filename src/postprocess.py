import csv
import os
import json
import sys

import cv2
import skimage.io
import numpy as np

def get_detection_grid(image_path, detections):
    image = skimage.io.imread(image_path)
    height, width = image.shape[:2]
    height = int(np.ceil(height / 100))
    width = int(np.ceil(width / 100))

    grid = [[0] * width for i in range(height)]
    for i in range(height):
        for j in range(width):
            grid[i][j] = []

    for detect in detections:
        orig_coords = detect['coords']

        i = int(orig_coords[0]) // 100
        j = int(orig_coords[1]) // 100

        grid[i][j].append(int(detect['label']))

    return grid, height, width


def dump_submission(image_path, detections, output_file):
    grid, height, width = get_detection_grid(image_path, detections)

    atlas_id = os.path.splitext(os.path.basename(image_path))[0]
    with open(output_file, 'a') as file:
        csv_writer = csv.writer(file)
        for i in range(height):
            for j in range(width):
                tile_id = f'{atlas_id}_{j*100}_{i*100}'
                if len(grid[i][j]):
                    labels = ' '.join(str(label) for label in sorted(grid[i][j]))
                else:
                    labels = ''

                csv_writer.writerows([[tile_id] + [labels]])


def process_file(input_file, detection_file, output_file):
    with open(detection_file) as f:
        detections = json.load(f)

        dump_submission(input_file, detections, output_file)
