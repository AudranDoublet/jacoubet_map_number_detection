import os
import json
import sys

import skimage.io
from PIL import ImageFont, ImageDraw, Image

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



def process_file(input_file, detection_file, output_file):
    print(f'Debug {input_file}')
    with open(detection_file) as f:
        detections = json.load(f)

        save_debug_image(input_file, detections, output_file)
