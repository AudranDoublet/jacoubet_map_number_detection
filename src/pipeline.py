import os
import sys

import dewarp
import heatmaps
import rewarp
import segmentation

def check_already_done(output_file):
    return os.path.exists(output_file)

class DewarpStep:
    def run(self, input_file, output_file, force=False):
        """
        Dewarp input image `input_file` and write result in `output_file`
        """
        if force or not check_already_done(output_file):
            return dewarp.process_file(input_file, output_file)


    def run_pipeline(self, pipeline):
        pipeline.dewarp_matrix = self.run(
            pipeline.input_file(),
            pipeline.create_file("dewarp", "01_dewarped.png")
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


class HeatmapStep:
    def run(self, input_file, output_file, force=False):
        """
        Find heatmaps
        """
        if force or not check_already_done(output_file):
            heatmaps.process_file(input_file, output_file)


    def run_pipeline(self, pipeline):
        self.run(
            pipeline.file("preprocessed"),
            pipeline.create_file("heatmaps", "03_heatmaps.png")
        )


class SegmentationStep:
    def run(self, input_file, output_file, force=False):
        """
        Segment heatmaps
        """
        segmentation.process_from_heatmaps(input_file, output_file)


    def run_pipeline(self, pipeline):
        self.run(
            pipeline.file("heatmaps"),
            pipeline.create_file("segments", "04_segments")
        )

class PostprocessingStep:
    def run(self, input_file, input_matrix, output_file, force=False):

        if force or not check_already_done(output_file):
            rewarp.process_file(input_file, input_matrix, output_file)


    def run_pipeline(self, pipeline):
        self.run(
            pipeline.file("segments"),
            pipeline.dewarp_matrix,
            pipeline.create_file("postprocessed", "05_submission.csv"),
        )


pipeline_steps = [
    ('Dewarp',        DewarpStep),
    ('Preprocessing', PreprocessingStep),
    ('Heatmaps',      HeatmapStep),
    ('Segmentation',  SegmentationStep),
    ('Postprocessing', PostprocessingStep),
]

class Pipeline:
    def __init__(self, input_file, output_dir):
        self._input_file = input_file
        self._output_dir = output_dir
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
    if len(sys.argv) != 3:
        print("Usage: pipeline.py <input_file> <output_dir>")

    else:
        Pipeline(sys.argv[1], sys.argv[2]).run()
