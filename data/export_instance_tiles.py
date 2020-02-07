from pathlib import Path
import argparse
import json
from math import inf
import multiprocessing as mp
from instance_tile_exporter import InstanceTileExporter


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', type=Path)
    parser.add_argument('experiment_name', type=str, default=None)
    parser.add_argument('--tile_size', type=int, default=1024)
    parser.add_argument('--mpp', type=float, default=0.4)
    parser.add_argument('--outer_label', type=str, default='epithelium')
    parser.add_argument('--label_values', type=json.loads, default='[["epithelium", 200], ["lumen", 250]]',
                        help='!!! NB: this would be "[[\"epithelium\", 200], [\"lumen\", 250]]" if passed externally')
    parser.add_argument('--roi_dir_name', default='tumour_area_annotations')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--max_instances_per_slide', type=int, default=inf)
    parser.add_argument('--tiles_dirname', type=str, default='tiles')
    parser.add_argument('--stop_overwrite', action='store_true')
    args = parser.parse_args()
    try:
        # get label directory
        dir_path = next(path for path in (args.data_dir/'data'/args.tiles_dirname).iterdir()
                        if path.is_dir() and args.outer_label in path.name)
    except (FileNotFoundError, StopIteration):
        dir_path = None

    def run_exporter(slide_id):
        if args.stop_overwrite and dir_path is not None and not (dir_path/slide_id).is_dir():
            return
        exporter = InstanceTileExporter(args.data_dir,
                                        slide_id,
                                        experiment_name=args.experiment_name,
                                        tile_size=args.tile_size,
                                        mpp=args.mpp,
                                        label_values=args.label_values)
        exporter.export_tiles(args.outer_label, args.data_dir / 'data' / args.tiles_dirname,
                              max_instances=args.max_instances_per_slide)
        print(f"Instance tiles exported from {slide_id}")

    slide_ids = [annotation_path.with_suffix('').name
                 for annotation_path in (args.data_dir/'data'/'annotations'/args.experiment_name).iterdir()
                 if annotation_path.suffix == '.json']

    if args.workers:
        with mp.Pool(args.workers) as pool:
            pool.map(run_exporter, slide_ids)
    else:
        for slide_id in slide_ids:
            run_exporter(slide_id)
    print("Done!")



