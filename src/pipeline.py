import os
import sys
import structlog

import dewarp
import grid_detection
import road_segmentation
import heatmaps
import postprocess
import segmentation
import labeller

logger = structlog.get_logger()

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
            model_path=pipeline.global_file("model")
        )

class PostprocessingStep:
    def run(self, input_file, detection_file, dewarp_matrix_file, output_file, force=False):
        postprocess.process_file(input_file, detection_file, dewarp_matrix_file, output_file)

    def run_pipeline(self, pipeline):
        self.run(
            pipeline.input_file(),
            pipeline.file("labels"),
            pipeline.file("dewarp_matrix"),
            pipeline.global_file("submission"),
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
        if os.path.isdir(input_file):
            self._input_files = [os.path.join(input_file, f) for f in os.listdir(input_file)]
        else:
            self._input_files = [input_file]

        self._current_file = 0
        self._output_dir = output_dir

        self._known_files = {}
        self._known_global_files = {
            "model": model_path,
            "submission": os.path.join(output_dir, "submission.json"),
        }

        self.dewarp_matrix = None


    def input_file(self):
        return self._input_files[self._current_file]

    def file_output_dir(self):
        return os.path.join(self._output_dir, f"{self._current_file:03}")


    def create_file(self, idx, name):
        self._known_files[idx] = os.path.join(self.file_output_dir(), name)
        return self.file(idx)


    def file(self, idx):
        return self._known_files[idx]


    def global_file(self, idx):
        return self._known_global_files[idx]


    def _run_file(self, idx, from_step):
        log = logger.new(file=self._input_files[idx])

        self._current_file = idx
        self._known_files = {}

        os.makedirs(self.file_output_dir(), exist_ok=True)

        for step in range(from_step, len(pipeline_steps)):
            (name, impl) = pipeline_steps[step]
            log.info("run_step", step=name)

            impl().run_pipeline(self)


    def run(self, from_step=None):
        os.makedirs(self._output_dir, exist_ok=True)

        i = 0

        if from_step is not None:
            for (j, (step, _)) in enumerate(pipeline_steps):
                if step.lower() == step.lower():
                    i = j
                    break

        for idx in range(len(self._input_files)):
            self._run_file(idx, i)


if __name__ == '__main__':
    if len(sys.argv) != 4:
        print("Usage: pipeline.py <input_file> <output_dir> <model_path>")

    else:
        Pipeline(sys.argv[1], sys.argv[2], sys.argv[3]).run()
