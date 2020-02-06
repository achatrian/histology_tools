import argparse
from pathlib import Path
from annotation_builder import AnnotationBuilder
from tqdm import tqdm
import cv2


r"""
Filter annotations by different contour statistics. 
E.g. size of contours
Moment disparity (for very long lines)

"""


def size_threshold(contour, min_area=1000.0):
    area = cv2.contourArea(contour)
    return area > min_area


def contour_ratio(contour, max_moment_ratio=1000.0):
    m = cv2.moments(contour)
    mx2, my2 = m['m20'], m['m02']
    return mx2/my2 < max_moment_ratio or my2/mx2 < max_moment_ratio


filters = (size_threshold, contour_ratio)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', type=Path)
    parser.add_argument('experiment_name', type=str)
    parser.add_argument('--layer_name', type=str, default='epithelium')
    parser.add_argument('--min_area', type=float, default=1000.0)
    parser.add_argument('--max_moment_ratio', type=float, default=1000.0)

    args = parser.parse_args()
    annotations_dir = args.data_dir/'data'/'annotations'/args.experiment_name
    annotation_paths = tuple(path for path in annotations_dir.iterdir() if path.suffix == '.json')
    for path in tqdm(annotation_paths):
        annotation = AnnotationBuilder.from_annotation_path(path)
        annotation.filter(args.layer_name, filters)
        # TODO finish and test

