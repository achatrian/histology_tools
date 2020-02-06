import argparse
from pathlib import Path
import re
from tqdm import tqdm
from annotation.annotation_builder import AnnotationBuilder

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('review_dir', type=Path)
    parser.add_argument('save_dir', type=Path)
    parser.add_argument('--num_splits', type=int)
    args = parser.parse_args()
    split_pattern = re.compile(r'_\d\.json')
    slide_ids = set()
    split_counts = dict()
    for path in args.review_dir.iterdir():
        match = split_pattern.search(path.name)
        if match is None:
            continue
        slide_id = path.name.replace(match.group(), '')  # remove split number
        slide_ids.add(slide_id)
        if slide_id not in split_counts:
            split_counts[slide_id] = 0
        split_counts[slide_id] += 1
    args.save_dir.mkdir(exist_ok=True, parents=False)
    for slide_id in tqdm(slide_ids):
        if split_counts[slide_id] != args.num_splits:
            raise FileNotFoundError(f"There are only {split_counts[slide_id]} split files for '{slide_id}' ({args.num_splits} were expected) in {str(args.review_dir)}")
        merged = AnnotationBuilder.merge(args.review_dir, slide_id)
        merged.dump_to_json(args.save_dir)
    print(f"{len(slide_ids)} merged annotations were saved!")

