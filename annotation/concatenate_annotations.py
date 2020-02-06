import argparse
from pathlib import Path
from tqdm import tqdm
from annotation.annotation_builder import AnnotationBuilder


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('annotation_dir0', type=Path)
    parser.add_argument('annotation_dir1', type=Path)
    parser.add_argument('save_dir', type=Path)
    opt = parser.parse_args()
    annotations_paths0, annotations_paths1 = (list(path for path in opt.annotation_dir0.iterdir() if path.suffix == '.json'),
                                  list(path for path in opt.annotation_dir1.iterdir() if path.suffix == '.json'))
    common_slide_ids = set(path.with_suffix('').name for path in annotations_paths0).intersection(
        set(path.with_suffix('').name for path in annotations_paths1)
    )
    opt.save_dir.mkdir(exist_ok=True)
    print(f"Merging annotations in {str(opt.annotation_dir0)} and {str(opt.annotation_dir1)} and saving to {str(opt.save_dir)}")
    for slide_id in tqdm(common_slide_ids):
        annotation_path0 = next(path for path in annotations_paths0 if slide_id in path.name)
        annotation_path1 = next(path for path in annotations_paths1 if slide_id in path.name)
        annotation0 = AnnotationBuilder.from_annotation_path(annotation_path0)
        annotation1 = AnnotationBuilder.from_annotation_path(annotation_path1)
        concatenated = AnnotationBuilder.concatenate(annotation0, annotation1)
        concatenated.dump_to_json(opt.save_dir)
        annotations_paths0.remove(annotation_path0), annotations_paths1.remove(annotation_path1)
    print("Done!")



