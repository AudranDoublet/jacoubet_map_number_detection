import os
import sys
import structlog
import shutil

import preprocessing
import grid_detection
import road_segmentation
import heatmaps
import postprocess
import segmentation
import labeller
import debug
import merge

logger = structlog.get_logger()

def check_already_done(output_file):
    return os.path.exists(output_file)


def check_already_done_all(files):
    return any([not check_already_done(f) for f in files])


class GenericStep:
    ALWAYS_RUN = False
    DEBUG_STEP = False

    def impl(self, *kargs):
        """
        Step implementation
        Should be overriden by child class
        """
        pass

    def input_files(self, pipeline):
        """
        Input files of step
        May be overriden by child class
        """
        return []


    def output_files(self, pipeline):
        """
        Output files of step
        May be overriden by child class
        """
        return []


    def output_folders(self, pipeline):
        """
        Output folders of step
        May be overriden by child class
        """
        return []


    def run_pipeline(self, pipeline, force=False):
        input_files = self.input_files(pipeline)
        output_files = self.output_files(pipeline)
        output_folders = self.output_folders(pipeline)

        should_run = self.ALWAYS_RUN or force

        # if an output file/folder doesn't exists, we should run
        if check_already_done_all(output_files + output_folders):
            should_run = True

        # run
        if should_run:
            # delete existing output folders
            for f in output_folders:
                if check_already_done(f):
                    shutil.rmtree(f)

            args = input_files + output_files + output_folders
            self.impl(*args)

            # update next steps if:
            # * this step has and output file
            # * and this step is not marked as debug
            return (not self.DEBUG_STEP) and len(output_files + output_folders) > 0

        return False


class GridDetectionStep(GenericStep):
    def impl(self, *kargs):
        grid_detection.process_image(*kargs)


    def input_files(self, pipeline):
        return [
            pipeline.input_file(),
        ]


    def output_files(self, pipeline):
        return [
            pipeline.create_file("exterior", "01_exterior.png"),
            pipeline.create_file("grid", "01_grid.png"),
        ]


class PreprocessingStep(GenericStep):
    def impl(self, *kargs):
        preprocessing.process_file(*kargs)


    def input_files(self, pipeline):
        return [
            pipeline.input_file(),
            pipeline.file("exterior"),
        ]


    def output_files(self, pipeline):
        return [
            pipeline.create_file("preprocessed", "02_preprocessed.png"),
        ]


class RoadSegmentation(GenericStep):
    def impl(self, *kargs):
        road_segmentation.process_file(*kargs)


    def input_files(self, pipeline):
        return [
            pipeline.file("preprocessed"),
            pipeline.file("grid"),
            pipeline.file("exterior"),
        ]


    def output_files(self, pipeline):
        return [
            pipeline.create_file("roads", "02_road_mask.png"),
        ]


class HeatmapStep(GenericStep):
    def impl(self, *kargs):
        heatmaps.process_file(*kargs)


    def input_files(self, pipeline):
        return [
            pipeline.file("preprocessed"),
            pipeline.file("roads"),
            pipeline.file("grid"),
            pipeline.file("exterior"),
        ]


    def output_files(self, pipeline):
        return [
            pipeline.create_file("heatmaps", "03_heatmaps.png"),
        ]


class SegmentationStep(GenericStep):
    def impl(self, *kargs):
        segmentation.process_from_heatmaps(*kargs)


    def input_files(self, pipeline):
        return [
            pipeline.file("heatmaps"),
            pipeline.file("roads"),
        ]


    def output_folders(self, pipeline):
        return [
            pipeline.create_file("segments", "04_segments"),
        ]


class LabelingStep(GenericStep):
    def impl(self, *kargs):
        labeller.label_segments(*kargs)


    def input_files(self, pipeline):
        return [
            pipeline.file("segments"),
            pipeline.global_file("model"),
        ]


    def output_files(self, pipeline):
        return [
            pipeline.create_file("labels", "05_labels.json"),
        ]


class MergeStep(GenericStep):
    def impl(self, *kargs):
        merge.process(*kargs)


    def input_files(self, pipeline):
        return [
            pipeline.file("labels"),
        ]


    def output_files(self, pipeline):
        return [
            pipeline.create_file("merge_labels", "06_merged_labels.json"),
        ]


class PostprocessingStep(GenericStep):
    ALWAYS_RUN = True

    def impl(self, *kargs):
        postprocess.process_file(*kargs)


    def input_files(self, pipeline):
        return [
            pipeline.input_file(),
            pipeline.file("merge_labels"),
            pipeline.global_file("submission"),
        ]


class DebugStep(GenericStep):
    DEBUG_STEP = True

    def impl(self, *kargs):
        debug.process_file(*kargs)


    def input_files(self, pipeline):
        return [
            pipeline.input_file(),
            pipeline.file("merge_labels"),
        ]


    def output_files(self, pipeline):
        return [
            pipeline.create_file("debug", "06_debug.png"),
        ]


class DebugSegmentationStep(GenericStep):
    DEBUG_STEP = True

    def impl(self, *kargs):
        debug.show_boxes(*kargs)


    def input_files(self, pipeline):
        return [
            pipeline.file("heatmaps"),
            pipeline.file("segments"),
        ]


    def output_files(self, pipeline):
        return [
            pipeline.create_file("debug_seg", "04_debug.png"),
        ]


pipeline_steps = [
    ('GridDetection',     GridDetectionStep),
    ('Preprocessing',     PreprocessingStep),
    ('RoadSegmentation',  RoadSegmentation),
    ('Heatmaps',          HeatmapStep),
    ('Segmentation',      SegmentationStep),
    ('Labelization',      LabelingStep),
    ('MergeNumbers',      MergeStep),
    ('Postprocessing',    PostprocessingStep),
    ('Debug',             DebugStep),
    ('DebugSegmentation', DebugSegmentationStep),
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
            "submission": os.path.join(output_dir, "submission.csv"),
        }


    def prepare_submission_file(self):
        with open(self.global_file("submission"), "w") as f:
            f.write("ID,numero\n")


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
        force = False

        for step in range(from_step, len(pipeline_steps)):
            (name, impl) = pipeline_steps[step]
            log.info("run_step", step=name)

            if impl().run_pipeline(self, force=force):
                force = True


    def run(self, from_step=None):
        os.makedirs(self._output_dir, exist_ok=True)
        self.prepare_submission_file()

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
