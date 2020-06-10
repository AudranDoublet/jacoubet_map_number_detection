import os
import json

def save_submission(props, output_file):
    pass

def rewarp_coords(coords, rewarp_matrix):
    pass

def process_file(input_dir, input_matrix, output_file):
    segmentation_paths = [path for path in os.listdir(input_dir) if path.endswith('.json')]
    for path in segmentation_paths:
        with os.open(path) as file:
            prop = json.load(file)

