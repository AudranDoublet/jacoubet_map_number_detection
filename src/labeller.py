import keras
import json
import numpy as np
from skimage.io import imread
from skimage.transform import resize


from pathlib import Path

CST_BATCH_SIZE = 128
CST_IMAGE_SHAPE = (28,28)

def get_segments_paths(directory: Path):
    annotations = sorted(directory.glob("*.json"), key=lambda path: path.name.split('.')[0])
    segments = [path.parent / f"{path.name.split('.')[0]}.png" for path in annotations]
    assert(all(map(lambda ann: ann.exists(), segments))) # check that all segments exists

    return annotations, segments

def load_segments_images(segments_paths):
    result = []
    for path in segments_paths:
        image = imread(path)
        image = resize(image, CST_IMAGE_SHAPE)
        image = image.reshape(*CST_IMAGE_SHAPE, 1)
        result.append(image)
    return np.array(result)

def load_annotations(annotations_paths):
    result = []
    for p in annotations_paths:
        with p.open("r") as fs:
            result.append(json.load(fs))
    return result

def generate_result(predictions, annotations):
    result = []
    for i in range(len(predictions)):
        c = predictions[i]
        if c == 10: # skip noise
            continue

        (y0, x0, y1, x1) = annotations[i]["bbox"]

        result.append({
            "label": int(c),
            "coords": [y0+(y1-y0)//2, x0+(x1-x0)//2],
            "merge_id": annotations[i]["merge_id"],
            "bbox": annotations[i]["bbox"]
        })

    return result

def label_segments(directory_path, model_path, output_path):
    directory_path = Path(directory_path)
    output_path = Path(output_path)

    result = []

    # LOADING
    p_annotations, p_segments = get_segments_paths(directory_path)

    if len(p_segments) > 0: # FIXME: remove that horrible if
        annotations = load_annotations(p_annotations)
        segments = load_segments_images(p_segments)

        # PREDICTION
        model = keras.models.load_model(model_path)
        predictions = model.predict_classes(segments, batch_size=CST_BATCH_SIZE)

        assert(len(predictions) == len(annotations))

        result = generate_result(predictions, annotations)

    # DUMP RESULTS
    with output_path.open("w") as fs:
        json.dump(result, fs)

