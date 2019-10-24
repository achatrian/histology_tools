import argparse
from pathlib import Path
import imageio
from aida_convert.annotation_builder import AnnotationBuilder
from aida_convert.mask_converter import MaskConverter

r"""Example creation of AIDA annotation from 3 classes"""
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=Path)
parser.add_argument('--extension', default='png', choice=['png', 'jpg'])
args, unparsed = parser.parse_known_args()


ground_truths = [imageio.imread(path) for path in args.data_dir.glob(f'*.{args.extension}')]
offsets = []

# Hierarchy of values, from background to foreground, in image
value_hier = (0, 200, 250)
# Descriptive value for each class
label_value_map = {
                'epithelium': 200,
                'lumen': 250,
                'background': 0
}
# Interval value for each class (i.e. if ground truth is not perfect)
label_interval_map = {
            'epithelium': (31, 225),
            'lumen': (225, 250),
            'background': (0, 30)
}

# Options to process each label:
label_options = {
            'epithelium': {
                'small_object_size': 1024*0.4,
                'dist_threshold': 0.05,
                'final_closing_size': 20,
                'final_dilation_size': 2
            },
            'lumen': {
                'small_object_size': 400,
                'dist_threshold': 0.001,
                'final_closing_size': 15,
                'final_dilation_size': 5
            }
}

converter = MaskConverter(
    value_hier=value_hier,
    label_value_map=label_value_map,
    label_interval_map=label_interval_map,
    label_options=label_options
)

my_classes = ['epithelium', 'lumen', 'background']
annotation = AnnotationBuilder('slide_1', 'my_project', my_classes)

# add iteratively all the contours to the annotation as labelled items
for ground_truth, offset in zip(ground_truths, offsets):
    contours, labels, boxes = converter.mask_to_contour(ground_truth, offset[0], offset[1])
    for contour, label, box in zip(contours, labels, boxes):
        annotation.add_item(label, 'path')
        contour = contour.squeeze().astype(int).tolist()  # deal with extra dim at pos 1
        annotation.add_segments_to_last_item(contour)

# save into json file
annotation.dump_to_json(args.data_dir)
