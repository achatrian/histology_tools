import argparse
from pathlib import Path
from tqdm import tqdm
from annotation.annotation_builder import AnnotationBuilder

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('annotation_dir', type=Path)
    parser.add_argument('n', type=int)
    parser.add_argument('--mode', type=str, default='parts')
    parser.add_argument('--roi_layer', type=str, default='Tumour area')
    parser.add_argument('-s', '--slide_ids', action='append', default=[])
    opt = parser.parse_args()
    annotation_paths = tuple(path for path in opt.annotation_dir.iterdir() if path.suffix == '.json')
    if opt.slide_ids:
        annotation_paths = [next(path for path in annotation_paths
                            if any(slide_id in path.name for slide_id in opt.slide_ids))]
    for annotation_path in tqdm(annotation_paths):
        annotation = AnnotationBuilder.from_annotation_path(annotation_path)
        splits = annotation.split(opt.mode, opt.n, roi_layer=opt.roi_layer, save_dir=annotation_path.parent)
        tqdm.write(f"{annotation_path.name} was split into {len(splits)} parts")
    print("Done!")
