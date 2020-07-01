import os
import json
import sys

from pathlib import Path

class Label:
    def __init__(self, label=None, coords=None, bbox=None, merge_id=None, json_obj=None):
        if json_obj:
            self.label = json_obj["label"]
            self.coords = json_obj["coords"]
            self.bbox = json_obj["bbox"]
            self.merge_id = json_obj["merge_id"]
        else:
            self.label = label
            self.coords = coords
            self.bbox = bbox
            self.merge_id = merge_id

    def to_json(self):
        return {
            "label": self.label,
            "coords": self.coords,
            "bbox": self.bbox,
            "merge_id": self.merge_id,
        }


def get_labels(filepath):
    """
    Read label json and store the objects in dict, key = merge_id, values = List(Label)
    """
    with open(filepath, "r") as f:
        labels = json.load(f)

        label_dict = {}

        for label_obj in labels:
            if label_dict.get(label_obj["merge_id"]):
                label_dict[label_obj["merge_id"]].append(Label(json_obj=label_obj))
            else:
                label_dict[label_obj["merge_id"]] = [Label(json_obj=label_obj)]

        return label_dict

    return None

def save_labels(filepath, labels):
    """
    labels = list(Label)
    """
    labels_json = [obj.to_json() for obj in labels]

    with open(filepath, "w") as fs:
        json.dump(labels_json, fs)

def merge_labels(label_list):
    """
    have the same merge_id
    """
    if len(label_list) == 1:
        return label_list[0] # single number

    print(f"merged id: {label_list[0].merge_id}")

    def create_number(label_list):
        res = 0
        for l in label_list:
            res = res * 10 + l.label
        return res

    def merge_bbox(label_list):
        min_y, min_x, max_y, max_x = label_list[0].bbox

        for i in range(1, len(label_list)):
            min_y = min(min_y, label_list[i].bbox[0])
            min_x = min(min_x, label_list[i].bbox[1])
            max_y = max(max_y, label_list[i].bbox[2])
            max_x = max(max_x, label_list[i].bbox[3])

        return [min_y, min_x, max_y, max_x]

    # sort by id_object, the first one is on the left
    labels = sorted(label_list, key=lambda label: label.bbox[1]) # x min

    new_bbox = merge_bbox(labels)

    new_label = Label(
        create_number(labels),
        [
            new_bbox[0] + (new_bbox[2] - new_bbox[0]) // 2,
            new_bbox[1] + (new_bbox[3] - new_bbox[1]) // 2,
        ],
        new_bbox,
        labels[0].merge_id
    )

    return new_label

def process(label_file, output_file):
    """
    Merge the figures into numbers
    """
    print(f'Merge {output_file}')

    labels = get_labels(label_file)

    result = [merge_labels(label_list) for label_list in labels.values()]

    save_labels(output_file, result)

if __name__ == "__main__":
    process(sys.argv[1], sys.argv[2])
