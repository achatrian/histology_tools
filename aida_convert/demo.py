import argparse
from pathlib import Path
import imageio
from .annotation_builder import AnnotationBuilder
from .mask_converter import mask_converter

r"""Example creation of AIDA annotation from 3 classes"""
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=Path)
parser.add_argument('--extension', default='png', choice=['png', 'jpg'])
args, unparsed = parser.parse_known_arguments()


ground_truths = [imageio.imread(path) for path in args.data_dir.glob(f'*.{args.extension}')]
offsets = []

# Hierarchy of values, from background to foreground, in image
value_hier = (0, (160, 200), 250)
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

converter = MaskConverter(
        value_hier=value_hier,
        label_value_map=label_value_map,
        label_interval_map=label_interval_map,
        min_contour_area=3000  # contours smaller than this area will be discarded
    )

my_classes = ['epithelium', 'lumen', 'background']
annotation = AnnotationBuilder('slide_1', 'my_project', my_classes)

# add iteratively all the contours to the annotation as labelled items
for ground_truth, offset in zip(ground_truths, offsets):
    contours, labels, boxes = converter.mask_to_contour(ground_truth, offset[0], offset[1])
    for contour, label, box in zip(contours, labels, boxes):
        annotation.add_item(label, 'path', tile_rect=box)
        contour = contour.squeeze().astype(int).tolist()  # deal with extra dim at pos 1
        annotation.add_segments_to_last_item(contour)

# Merge overlapping segments of the same class
annotation.merge_overlapping_segments(parallel=True, num_workers=4, log_dir=args.data_dir)
annotation.dump_to_json(args.data_dir)
