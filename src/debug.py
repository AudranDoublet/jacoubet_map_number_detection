import os
import json
import sys

import skimage.io
from PIL import ImageFont, ImageDraw, Image
from pathlib import Path

def save_debug_image(image_path, detections, output_file):
    img = Image.open(image_path)
    draw = ImageDraw.Draw(img)

    fonts_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'fonts')
    font_file = os.path.join(fonts_path, "Neufreit-ExtraBold.otf")
    font = ImageFont.truetype(font_file, 20)
    color_rgb = (0, 100, 150)
    offset = (5, -30)

    for detect in detections:
        bbox = detect['bbox']
        box = [bbox[1], bbox[0], bbox[3], bbox[2]]
        draw.rectangle(box, outline=(255, 0, 0))
        draw.text(
            (bbox[3] + offset[0], bbox[0] + offset[1]), # (x, y)
            str(detect["label"]),
            color_rgb,
            font=font
        )

    img.save(output_file)

# At the end of the pipeline
def process_file(input_file, detection_file, output_file):
    """
    For each object: draw rectangles around bbox + show the result of algo
    """
    with open(detection_file) as f:
        detections = json.load(f)

        save_debug_image(input_file, detections, output_file)


# After segmentation part
def show_boxes(input_file, detection_directory, output_file):
    """
    Draw rectangles for each bbox objects, after segementation step
    """

    def get_annotation_files(directory):
        return sorted(
            directory.glob("*.json"),
            key=lambda path: path.name.split('.')[0]
        )

    def get_bbox(annotation_path):
        with annotation_path.open("r") as f:
            return json.load(f)["bbox"]

    fonts_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'fonts')
    font_file = os.path.join(fonts_path, "Neufreit-ExtraBold.otf")
    font = ImageFont.truetype(font_file, 16)
    color_rgb = (0, 100, 150)
    offset = (5, -30)

    img = Image.open(input_file)
    draw = ImageDraw.Draw(img)

    for annot in get_annotation_files(Path(detection_directory)):
        bbox = get_bbox(annot)
        box = [bbox[1], bbox[0], bbox[3], bbox[2]]
        draw.rectangle(box, outline=150)
        draw.text(
            (bbox[3] + offset[0], bbox[0] + offset[1]), # (x, y)
            str(os.path.basename(annot)).split('.')[0],
            color_rgb,
            font=font
        )

    img.save(output_file)
