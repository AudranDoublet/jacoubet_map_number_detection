import os
import sys

import dewarp
import grid_detection
import road_segmentation
import heatmaps
import postprocess
import segmentation
import labeller

def check_already_done(output_file):
    return os.path.exists(output_file)

class DewarpStep:
    def run(self, input_file, output_file, matrix_output_file, force=False):
        """
        Dewarp input image `input_file` and write result in `output_file`
        """
        if force or not check_already_done(output_file):
            dewarp.process_file(input_file, output_file, matrix_output_file)


    def run_pipeline(self, pipeline):
        self.run(
            pipeline.input_file(),
            pipeline.create_file("dewarp", "01_dewarped.png"),
            pipeline.create_file("dewarp_matrix", "01_dewarp_matrix.csv")
        )


class GridDetectionStep:
    def run(self, input_file, output_file, force=False):
        """
        Dewarp input image `input_file` and write result in `output_file`
        """
        if force or not check_already_done(output_file):
            grid_detection.process_file(input_file, output_file)


    def run_pipeline(self, pipeline):
        self.run(
            pipeline.file("dewarp"),
            pipeline.create_file("grid", "02_grid.png")
        )


class PreprocessingStep:
    def run(self, input_file, output_file, force=False):
        import shutil
        shutil.copy(input_file, output_file) # FIXME


    def run_pipeline(self, pipeline):
        self.run(
            pipeline.file("dewarp"),
            pipeline.create_file("preprocessed", "02_preprocessed.png")
        )


class RoadSegmentation:
    def run(self, input_file, grid_file, output_file, force=False):
        if force or not check_already_done(output_file):
            road_segmentation.process_file(input_file, grid_file, output_file)


    def run_pipeline(self, pipeline):
        self.run(
            pipeline.file("preprocessed"),
            pipeline.file("grid"),
            pipeline.create_file("roads", "02_road_mask.png")
        )


class HeatmapStep:
    def run(self, input_file, road_file, output_file, force=False):
        """
        Find heatmaps
        """
        if force or not check_already_done(output_file):
            heatmaps.process_file(input_file, road_file, output_file)


    def run_pipeline(self, pipeline):
        self.run(
            pipeline.file("preprocessed"),
            pipeline.file("roads"),
            pipeline.create_file("heatmaps", "03_heatmaps.png")
        )


class SegmentationStep:
    def run(self, input_file, road_file, output_file, force=False):
        """
        Segment heatmaps
        """
        segmentation.process_from_heatmaps(input_file, road_file, output_file)


    def run_pipeline(self, pipeline):
        self.run(
            pipeline.file("heatmaps"),
            pipeline.file("roads"),
            pipeline.create_file("segments", "04_segments")
        )

class LabelingStep:
    def run(self, input_directory, output_file, model_path, force=False):
        labeller.label_segments(input_directory, output_file, model_path)

    def run_pipeline(self, pipeline):
        self.run(
            input_directory=pipeline.file("segments"),
            output_file=pipeline.create_file("labels", "05_labels.json"),
            model_path=pipeline.model_path
        )

class PostprocessingStep:
    def run(self, input_file, detection_file, dewarp_matrix_file, output_file, force=False):
        postprocess.process_file(input_file, detection_file, dewarp_matrix_file, output_file)

    def run_pipeline(self, pipeline):
        self.run(
            pipeline._input_file,
            pipeline.file("labels"),
            pipeline.file("dewarp_matrix"),
            pipeline.create_file("postprocessed", "06_submission.csv"),
        )

pipeline_steps = [
    ('Dewarp',           DewarpStep),
    ('GridDetection',    GridDetectionStep),
    ('Preprocessing',    PreprocessingStep),
    ('RoadSegmentation', RoadSegmentation),
    ('Heatmaps',         HeatmapStep),
    ('Segmentation',     SegmentationStep),
    ('Labelization',  LabelingStep),
    ('Postprocessing',   PostprocessingStep),
]

class Pipeline:
    def __init__(self, input_file, output_dir, model_path):
        self._input_file = input_file
        self._output_dir = output_dir
        self.model_path = model_path
        self._known_files = {}

        self.dewarp_matrix = None

    def input_file(self):
        return self._input_file

    def create_file(self, idx, name):
        self._known_files[idx] = os.path.join(self._output_dir, name)
        return self.file(idx)

    def file(self, idx):
        return self._known_files[idx]

    def run(self, from_step=None):
        os.makedirs(self._output_dir, exist_ok=True)

        i = 0

        if from_step is not None:
            for (j, (step, _)) in enumerate(pipeline_steps):
                if step.lower() == step.lower():
                    i = j
                    break

        for step in range(i, len(pipeline_steps)):
            (_, impl) = pipeline_steps[step]

            impl().run_pipeline(self)


if __name__ == '__main__':
    if len(sys.argv) != 4:
        print("Usage: pipeline.py <input_file> <output_dir> <model_path>")

    else:
        Pipeline(sys.argv[1], sys.argv[2], sys.argv[3]).run()
