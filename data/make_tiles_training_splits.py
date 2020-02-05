import argparse
from pathlib import Path
import json
from datetime import datetime
from sklearn.model_selection import KFold


# does not take any labels into account
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', type=Path)
    parser.add_argument('--roi_layer', default='Tumour area')
    parser.add_argument('--n_splits', type=int, default=4)
    args = parser.parse_args()
    tiles_dir = args.data_dir/'data'/'tiles'/args.roi_layer
    if not tiles_dir.exists():
        raise FileNotFoundError(f"No tiles directory in {str(args.data_dir)}")
    slide_ids = tuple(path.name for path in tiles_dir.iterdir() if path.is_dir())
    splits = tuple(KFold(n_splits=args.n_splits).split(slide_ids))
    (args.data_dir/'data'/'CVsplits').mkdir(exist_ok=True)
    with open(args.data_dir/'data'/'CVsplits'/'tiles_split.json', 'w') as tiles_split_file:
        tile_splits = {
            'date': str(datetime.now()),
            'n_splits': args.n_splits,
            'roi_layer': args.roi_layer,
            'train': [],
            'test': []
        }
        for i, split in enumerate(splits):
            train_split_slides = tuple(slide_ids[j] for j in split[0])
            test_split_slides = tuple(slide_ids[j] for j in split[1])
            tile_splits['train'].append(train_split_slides)
            tile_splits['test'].append(test_split_slides)
        json.dump(tile_splits, tiles_split_file)
    print("Done!")



