from pathlib import Path
import argparse
import json
import multiprocessing as mp
import numpy as np
from roi_tile_exporter import ROITileExporter


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', type=Path)
    parser.add_argument('--tile_size', type=int, default=1024)
    parser.add_argument('--mpp', type=float, default=0.4)
    parser.add_argument('--max_num_tiles', default=np.int)
    parser.add_argument('--area_label', type=str, default='Tumour area')
    parser.add_argument('--label_values', type=json.loads, default='[["epithelium", 200], ["lumen", 250]]',
                        help='!!! NB: this would be "[[\"epithelium\", 200], [\"lumen\", 250]]" if passed externally')
    parser.add_argument('--roi_dir_name', default='tumour_area_annotations')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--stop_overwrite', action='store_true')
    args = parser.parse_args()
    try:
        # get label directory
        dir_path = next(path for path in (args.data_dir/'data'/'tiles').iterdir()
                        if path.is_dir() and args.outer_label in path.name)
    except StopIteration:
        dir_path = None

    def run_exporter(slide_id):
        if args.stop_overwrite and dir_path is not None and not (dir_path/slide_id).is_dir():
            return
        exporter = ROITileExporter(args.data_dir,
                                   slide_id,
                                   args.tile_size,
                                   args.mpp,
                                   roi_dir_name=args.roi_dir_name)
        exporter.export_tiles(args.area_label, args.data_dir / 'data' / 'tiles')
        print(f"ROI tiles exported from {slide_id}")

    slide_ids = [annotation_path.with_suffix('').name
                 for annotation_path in (args.data_dir/'data'/'annotations').iterdir()
                 if annotation_path.suffix == '.json']

    if args.workers:
        with mp.Pool(args.workers) as pool:
            pool.map(run_exporter, slide_ids)
    else:
        for slide_id in slide_ids:
            run_exporter(slide_id)
    print("Done!")



