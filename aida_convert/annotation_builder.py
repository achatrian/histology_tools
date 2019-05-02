import json
import math
import copy
import warnings
from collections import defaultdict, OrderedDict, namedtuple
from itertools import combinations, product
import logging
from datetime import datetime
import time
from pathlib import Path
import multiprocessing as mp
import queue
import tqdm
import numpy as np
from scipy.special import comb
import cv2


class AnnotationBuilder:
    """
    Use to create AIDA annotation .json files for a single image/WSI that can be read by AIDA
    """
    @classmethod
    def from_object(cls, obj):
        instance = cls(obj['slide_id'], obj['project_name'], obj['layer_names'])
        instance._obj = obj
        try:
            instance.metadata = obj['metadata']
        except KeyError:
            warnings.warn(f"No available metadata for {obj['slide_id']}")
        return instance
    
    def __init__(self, slide_id, project_name, layers=(), keep_original_paths=False):
        self.slide_id = slide_id  # AIDA uses filename before extension to determine which annotation to load
        self.project_name = project_name
        self._obj = {
            'name': self.project_name,
            'layers': []
        }  # to be dumped to .json
        self.layers = []
        # store additional info on segments for processing
        self.metadata = defaultdict(lambda: {'tile_dict': [], 'dist': []})  # layer_idx -> (metadata_name -> (item_idx -> value)))
        self.last_added_item = None
        # point comparison
        self.keep_original_paths = keep_original_paths
        self.merged = False  # if merge_overlapping_segments() is called and successfully completed, merged is turned to true
        for layer_name in layers:
            self.add_layer(layer_name)  # this also updates self.layers with references to layers names

    @staticmethod
    def euclidean_dist(p1, p2):
        x1, y1 = (p1['x'], p1['y']) if isinstance(p1, dict) else p1
        x2, y2 = (p2['x'], p2['y']) if isinstance(p2, dict) else p2
        return math.sqrt(math.pow(x2 - x1, 2) + math.pow(y2 - y1, 2))

    def __getitem__(self, idx):
        return self._obj['layers'][idx]

    def __setitem__(self, idx, value):
        self._obj['layers'][idx] = value

    def num_layers(self):
        return len(self._obj['layers'])

    def layer_has_items(self, layer_idx):
        if type(layer_idx) is str:
            layer_idx = self.get_layer_idx(layer_idx)  # if name is given rather than index
        layers_len = len(self._obj['layers'][layer_idx])
        if layer_idx > layers_len:
            raise IndexError(f'Index {layer_idx} is out of range for layers list with len {layers_len}')
        layer = self._obj['layers'][layer_idx]
        return bool(layer['items'])

    def get_layer_idx(self, layer_name):
        try:
            # turn into numerical index
            layer_idx = next(i for i, layer in enumerate(self._obj['layers']) if layer['name'] == layer_name)
        except StopIteration:
            raise ValueError(f"No layer with specified name {layer_name}")
        return layer_idx

    def add_layer(self, layer_name):
        new_layer = {
            'name': layer_name,
            'opacity': 1,
            'items': []
        }
        self._obj['layers'].append(new_layer)
        self.layers.append(self._obj['layers'][-1]['name'])
        self.metadata[self.layers[-1]]['tile_rect'] = []
        return self

    def add_item(self, layer_idx, type_, class_=None, points=None, tile_rect=None):
        r"""
        Add item to desired layer
        :param layer_idx: numerical index or layer name (converted to index)
        :param type_:
        :param class_:
        :param points:
        :param tile_rect:
        :return:
        """
        if type(layer_idx) is str:
            layer_idx = self.get_layer_idx(layer_idx)  # if name is given rather than index
        new_item = {
            'class': class_ if class_ else self._obj['layers'][layer_idx]['name'],
            'type': type_,
            "segments": [],
            'closed': True
        }  # TODO properties must change with item type - e.g. circle, rectangle
        self._obj['layers'][layer_idx]['items'].append(new_item)
        self.last_added_item = self._obj['layers'][layer_idx]['items'][-1]  # use to update item
        if points:
            self.add_segments_to_last_item(points)
        if tile_rect:
            # add tile info to metadata
            if not isinstance(tile_rect, (tuple, list)) or len(tile_rect) != 4:
                raise ValueError("Invalid tile rect was passed - must be a tuple (x, y, w, h) specifying the bounding box around the item's segments")
            self.metadata[self.layers[layer_idx]]['tile_rect'].append(tile_rect)
        return self

    def remove_item(self, layer_idx, item_idx):
        layer_idx = self.get_layer_idx(layer_idx)  # FIXME slightly confusing - this is getting layer idx from name or doing nothing
        del self._obj['layers'][layer_idx]['items'][item_idx]
        layer_name = self._obj['layers'][layer_idx]['name']
        del self.metadata[layer_name]['tile_rect'][item_idx]

    def set_layer_items(self, layer_idx, items):
        self._obj['layers'][layer_idx]['items'] = items
        return self

    def add_segments_to_last_item(self, points):
        segments = []
        for i, point in enumerate(points):
            x, y = point
            new_segment = {
                'point': {
                    'x': x,
                    'y': y,
                }
              }
            segments.append(new_segment)
        self.last_added_item['segments'] = segments
        return self

    def merge_overlapping_segments(self, closeness_thresh=5.0, dissimilarity_thresh=4.0, max_iter=1,
                                   parallel=True, num_workers=4, log_dir=''):
        """
        Compares all segments and merges overlapping ones
        """
        if parallel:
            self.parallel_merge_overlapping_segments(closeness_thresh, dissimilarity_thresh, max_iter, num_workers, log_dir=log_dir)
        else:
            # Set up logging
            logger = logging.getLogger(__name__)
            logger.setLevel(logging.DEBUG)
            fh = logging.FileHandler(Path(log_dir if log_dir else '.')/f'merge_segments_{datetime.now()}.log')  # logging to file for debugging
            fh.setLevel(logging.DEBUG)
            ch = logging.StreamHandler()  # logging to console for general runtime info
            ch.setLevel(logging.DEBUG)  # if this is set to ERROR, then errors are not printed to log, and vice versa
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            fh.setFormatter(formatter)
            ch.setFormatter(formatter)
            logger.addHandler(fh)
            logger.addHandler(ch)
            logger.info("Begin segment merging ...")
            # Merge by layer first - this can removes some uncertainty as complete paths are usually better classified than interrupted ones
            for layer in self._obj['layers']:
                changed = True
                num_iters = 0
                items = layer['items']
                logger.info(f"[Layer '{layer['name']}'] Initial number of items is {len(items)}.")
                merged_num = 0  # keep count of change in elements number
                while changed and num_iters < max_iter:
                    # repeat with newly formed contours
                    items = layer['items']
                    if num_iters == 0:
                        logger.info(f"[Layer '{layer['name']}'] First iteration ...")
                    else:
                        logger.info(f"[Layer '{layer['name']}'] Iter #{num_iters} cycle merged {len(store_items) - len(items)}")
                    store_items = copy.deepcopy(layer['items'])  # copy to iterate over
                    to_remove = set() # store indices so that cycle is not repeated if either item has already been removed / discarded
                    for (i, item0), (j, item1) in tqdm.tqdm(combinations(enumerate(store_items), 2),
                                                            leave=False, total=comb(N=len(store_items), k=2)):
                        if i == j or i in to_remove or j in to_remove:
                            continue  # items are the same or items have already been processed
                        # Brute force approach, shapes are not matched perfectly, so if there are many points close to another
                        # point and distance dosesn't match the points 1-to-1 paths might be merged even though they don't
                        try:  # get bounding boxes for contours from annotation meta-data
                            tile_rect0 = self.metadata[layer['name']]['tile_rect'][i]
                            tile_rect1 = self.metadata[layer['name']]['tile_rect'][j]
                        except KeyError as keyerr:
                            raise KeyError(f"{keyerr.args[0]} - Unspecified metadata for layer {layer['name']}")
                        except IndexError:
                            raise IndexError(f"Item {i if i > j else j} is missing from layer {layer['name']} metadata")
                        rects_positions, origin_rect, rect_areas = self.check_relative_rect_positions(tile_rect0, tile_rect1)
                        if not rects_positions:
                            continue  # do nothing if items are not from touching/overlapping tiles
                        points_near, points_far = (tuple(self.item_points(item0 if origin_rect == 0 else item1)),
                                                   tuple(self.item_points(item0 if origin_rect == 1 else item1)))
                        # points_near stores the points of the top and/or leftmost contour
                        if rects_positions == 'overlap':
                            # if contours overlap, remove overlapping points from the bottom / rightmost contour
                            points_far = self.remove_overlapping_points(points_near, points_far)
                            # try:
                            #     points_far[0]
                            # except IndexError as err:
                            #     logging.error(str(err.args), exc_info=True)
                        assert points_far, "Some points must remain from this operation, or rects_positions should have been 'contained'"
                        total_min_dist = 0.0
                        # noinspection PyBroadException
                        try:
                            # find pairs of points, one in each contour, that may lie at boundary
                            closest_points, point_dist = self.find_closest_points(self.euclidean_dist, points_near, points_far, closeness_thresh)
                            if closest_points and len(closest_points) > 1:
                                total_min_dist += sum(
                                    min(point_dist[p0][p1] if (p0, p1) in closest_points else 0 for p1 in points_far)
                                    for p0 in points_near
                                )  # sum dist
                                if total_min_dist / len(closest_points) < dissimilarity_thresh:
                                    outer_points = self.get_merged(points_near, points_far, closest_points)
                                    # make bounding box for new contour
                                    x_out, y_out = (min(tile_rect0[0], tile_rect1[0]), min(tile_rect0[1], tile_rect1[1]))
                                    w_out = max(tile_rect0[0] + tile_rect0[2], tile_rect1[0] + tile_rect1[2]) - x_out  # max(x+w) - min(x)
                                    h_out = max(tile_rect0[1] + tile_rect0[3], tile_rect1[1] + tile_rect1[3]) - y_out  # max(x+w) - min(x)
                                    self.add_item(layer['name'], item0['type'], class_=item0['class'], points=outer_points,
                                                  tile_rect=(x_out, y_out, w_out, h_out))
                                    # logging.debug(f"Item {i} and {j} were merged - average dist per close point = {total_min_dist / len(closest_points)} (threshold = {dissimilarity_thresh})")
                                    to_remove.add(i)
                                    to_remove.add(j)
                        except Exception:
                            logger.error(f"""
                            [Layer '{layer['name']}'] iter: #{num_iters} items: {(i, j)} total item num: {len(items)} merged items:{len(store_items) - len(items)}
                            [Bounding box] 0 x: {tile_rect0[0]} y: {tile_rect0[1]} w: {tile_rect0[2]} h: {tile_rect0[2]}
                            [Bounding box] 1 x: {tile_rect1[0]} y: {tile_rect1[1]} w: {tile_rect1[2]} h: {tile_rect1[2]}
                            Result of rectangle position check: '{rects_positions}', origin: {origin_rect}, areas: {rect_areas}
                            """)
                            logger.error('Failed.', exc_info=True)
                    for item_idx in sorted(to_remove, reverse=True):  # remove items that were merged - must start from last index or will return error
                        # noinspection PyBroadException
                        try:
                            self.remove_item(layer['name'], item_idx)
                        except Exception:
                            logger.error(f"Failed to remove item {item_idx} in layer {layer['name']}", exc_info=True)
                    changed = items != store_items
                    merged_num += len(store_items) - len(items)
                    num_iters += 1
                if changed:
                    logger.info(f"[Layer '{layer['name']}'] Max number of iterations reached ({num_iters})")
                else:
                    logger.info(f"[Layer '{layer['name']}'] No changes after {num_iters} iterations.")
                logger.info(f"[Layer '{layer['name']}'] {merged_num} total merged items.")
            self.merged = True
            logger.info('Done!')

    @staticmethod
    def find_closest_points(distance, points_near, points_far, closeness_thresh=3.0):
        """
        :param distance: distance metric taking two points and returning commutative value
        :param points_near:
        :param points_far:
        :param closeness_thresh:
        :return: pairs of corresponding points
                 distance between point pairs
        """
        point_dist = dict()  # point1 -> point2
        for (k, p0), (l, p1) in product(enumerate(points_near), enumerate(points_far)):
            # distance comparison
            try:
                point_dist[p0][p1] = distance(p0, p1)  # store distances
            except KeyError:
                point_dist[p0] = OrderedDict()
                point_dist[p0][p1] = distance(p0, p1)  # store distances
            if tuple(point_dist[p0].keys()) != points_far[0:l + 1]:  # must have same order as well as same elements
                assert tuple(point_dist[p0].keys()) == points_far[0:l + 1]  # must have same order as well as same elements
        closest_points = set()
        for p0 in points_near:
            closest_idx = np.argmin(list(point_dist[p0].values())).item()  # must list() value_view, as it is not a sequence
            closest_point = points_far[closest_idx]
            if point_dist[p0][closest_point] < closeness_thresh:
                closest_points.add((p0, points_far[closest_idx]))
        return closest_points, point_dist

    @staticmethod
    def get_merged(points_near, points_far, close_point_pairs, positions='horizontal'):
        """
        Function to merge contours from adjacent tiles
        :param points_near: leftmost (for positions = horizontal) or topmost (for position = vertical) path
        :param points_far:
        :param close_point_pairs:
        :param positions
        :return: merged points
        """
        close_point_pairs = tuple(close_point_pairs)
        # two-way mapping
        correspondance = {p0: p1 for p0, p1 in close_point_pairs}
        correspondance.update({p1: p0 for p0, p1 in close_point_pairs})
        close_points_near = set(p0 for p0, p1 in close_point_pairs)
        close_points_far = set(p1 for p0, p1 in close_point_pairs)
        # drop close segments
        outer_points = []
        if positions == '':
            raise ValueError("Cannot merge items if bounding boxes are not at least adjacent")
        # https://stackoverflow.com/questions/45323590/do-contours-returned-by-cvfindcontours-have-a-consistent-orientation
        # outer contours are oriented counter-clockwise
        # scanning for first point is done from top left to bottom right
        # assuming contours are extracted for each value independently
        for i0, point0 in enumerate(points_near):
            # start - lower extreme - higher - extreme
            outer_points.append(point0)
            if point0 in close_points_near and i0 != 0:  # p0 of lowest (p0, p1) pair
                break
        assert point0 in close_points_near, "This loop should end at a close point"
        start_p1 = correspondance[point0]
        start_p1_idx = points_far.index(start_p1)
        for i1, point1 in enumerate(points_far[start_p1_idx:] + points_far[:start_p1_idx]):
            outer_points.append(point1)
            if point1 in close_points_far and i1 != 0:  # p1 of highest (p0, p1) pair
                break
        assert point1 in close_points_far, "This loop should end at a close point"
        restart_p0 = correspondance[point1]
        for i00, point0 in enumerate(points_near[points_near.index(restart_p0):]):
            outer_points.append(point0)
            i0 += 1
        return outer_points

    @staticmethod
    def remove_overlapping_points(points_near, points_far):
        points_far = list(points_far)
        contour_near = np.array(points_near)[:, np.newaxis, :]
        for i1, point1 in reversed(list(enumerate(copy.copy(points_far)))):
            # cv2.pointPolygonTest
            # Positive value if the point is inside the contour !!!
            # Negative value if the point is outside the contour
            # Zero if the point is on the contour
            if cv2.pointPolygonTest(contour_near, point1, False) >= 0:
                del points_far[i1]
        return tuple(points_far)

    @staticmethod
    def item_points(item):
        """Generator over item's points"""
        # remove multiple identical points - as they can cause errors
        points = set()
        for segment in item['segments']:
            point = tuple(segment['point'].values())
            if point not in points:
                points.add(point)
                yield point

    @staticmethod
    def check_relative_rect_positions(tile_rect0, tile_rect1, eps=0):
        """
        :param tile_rect0:
        :param tile_rect1:
        :param eps: tolerance in checks
        :return: positions: overlap|horizontal|vertical|'' - the relative location of the two paths
                 origin_rect: meaning depends on relative positions of two boxes:
                            * contained: which box is biggerpng
                            * horizontal: leftmost box
                            * vertical: topmost box
                            * overlap: topmost box
                            * '': None
                 rect_areas: areas of the bounding boxes
        """
        x0, y0, w0, h0 = tile_rect0
        x1, y1, w1, h1 = tile_rect1
        x_w0, y_h0, x_w1, y_h1 = x0 + w0, y0 + h0, x1 + w1, y1 + h1
        # symmetric relationship - check only in one
        x_overlap = (x0 - eps <= x1 <= x_w0 + eps or x0 - eps <= x_w1 <= x_w0 + eps)  # no need for symmetric check
        y_overlap = (y0 - eps <= y1 <= y_h0 + eps or y0 - eps <= y_h1 <= y_h0 + eps)
        x_contained = (x0 - eps <= x1 <= x_w0 + eps and x0 - eps <= x_w1 <= x_w0 + eps) or \
                      (x1 - eps <= x0 <= x_w1 + eps and x1 - eps <= x_w0 <= x_w1 + eps)  # one is bigger than the other - not symmetric!
        y_contained = (y0 - eps <= y1 <= y_h0 + eps and y0 - eps <= y_h1 <= y_h0) or \
                      (y1 - eps <= y0 <= y_h1 + eps and y1 - eps <= y_h0 <= y_h1)
        if x_contained and y_contained:
            positions = 'contained'
            if (x_w0 - x0) * (y_h0 - y0) >= (x_w1 - x1) * (y_h1 - y1):
                origin_rect = 0  # which box is bigger
            else:
                origin_rect = 1
        elif not x_contained and y_overlap and (x_w0 < x1 + eps or x_w1 < x0 + eps):
            positions = 'horizontal'
            if x_w0 < x1 + eps:
                origin_rect = 0
            elif x_w1 < x0 + eps:
                origin_rect = 1
            else:
                raise ValueError("shouldn't be here")
        elif not y_contained and x_overlap and (y_h0 < y1 + eps or y_h1 < y0 + eps):
            positions = 'vertical'
            if y_h0 < y1 + eps:
                origin_rect = 0
            elif y_h1 < y0 + eps:
                origin_rect = 1
            else:
                raise ValueError("shouldn't be here")
        elif x_overlap and y_overlap:
            positions = 'overlap'
            if y0 <= y1:
                origin_rect = 0
            else:
                origin_rect = 1
        else:
            positions = ''
            origin_rect = None
        rect_areas = ((x_w0 - x0) * (y_h0 - y0), (x_w1 - x1) * (y_h1 - y1))
        return positions, origin_rect, rect_areas,

    def print(self, indent=4):
        print(json.dumps(self._obj, sort_keys=False, indent=indent))

    def export(self):
        """
        Add attributes so that obj can be used to create new data annotation
        :return:
        """
        obj = copy.deepcopy(self._obj)
        obj['project_name'] = self.project_name
        obj['slide_id'] = self.slide_id
        obj['layer_names'] = self.layers
        return obj, dict(self.metadata)  # defaultdict with lambda cannot be pickled

    def dump_to_json(self, save_dir, name='', suffix_to_remove=('.ndpi', '.svs')):
        save_path = Path(save_dir)/self.slide_id
        save_path = save_path.with_suffix('.json') if save_path.suffix in suffix_to_remove else \
            save_path.parent/(save_path.name +'.json')  # add json taking care of file ending in .some_text.[ext,no_ext]
        if name:
            save_path = str(save_path)[:-5] + '_' + name + '.json'
        obj, metadata = self.export()
        obj['metadata'] = metadata
        obj['merged'] = self.merged
        json.dump(obj, open(save_path, 'w'))

    def get_layer_points(self, layer_idx, contour_format=False):
        """Get all paths in a given layer, function used to extract layer from annotation object"""
        if isinstance(layer_idx, str):
            layer_idx = self.get_layer_idx(layer_idx)
        layer = self._obj['layers'][layer_idx]
        if contour_format:
            layer_points = list(
                np.array(list(self.item_points(item))).astype(np.int32)[:, np.newaxis, :]  # contour functions only work on int32
                if item['segments'] else np.array([]) for item in layer['items']
            )
        else:
            layer_points = list(list(self.item_points(item)) for item in layer['items'])
        return layer_points, layer['name']

    def parallel_merge_overlapping_segments(self, closeness_thresh=5.0, dissimilarity_thresh=4.0, max_iter=1,
                                            num_workers=4, log_dir='', timeout=60):
        """
        Compares all segments and merges overlapping ones
        """
        # Set up logging
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.DEBUG)
        fh = logging.FileHandler(Path(log_dir if log_dir else '.')/f'merge_segments_{datetime.now()}.log')  # logging to file for debugging
        fh.setLevel(logging.DEBUG)
        ch = logging.StreamHandler()  # logging to console for general runtime info
        ch.setLevel(logging.DEBUG)  # if this is set to ERROR, then errors are not printed to log, and vice versa
        formatter = logging.Formatter('%(asctime)s | %(levelname)s | ID: %(process)d | %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        logger.addHandler(fh)
        logger.addHandler(ch)
        funcs = Funcs(euclidean_dist=self.euclidean_dist,
                      check_relative_rect_positions=self.check_relative_rect_positions,
                      item_points=self.item_points,
                      remove_overlapping_points=self.remove_overlapping_points,
                      get_merged=self.get_merged,
                      find_closest_points=self.find_closest_points)  # to use functions in subprocess
        logger.info("Begin segment merging ...")
        layer_time = 0.0
        for r, layer in enumerate(self._obj['layers']):
            changed = True
            num_iters = 0
            items = layer['items']
            logger.info(f"[Layer '{layer['name']}'] Initial number of items is {len(items)}.")
            merged_num, num_put_to_process = 0, 0  # keep count of change in elements number
            layer_start_time = time.time()
            iter_time = 0.0
            while changed and num_iters < max_iter:
                iter_start_time = time.time()
                # repeat with newly formed contours
                items = layer['items']
                if num_iters == 0:
                    logger.info(f"[Layer '{layer['name']}'] Iter 0 begins ...")
                else:
                    logger.info(f"[Layer '{layer['name']}'] Iter #{num_iters} cycle merged {len(store_items) - len(items)} in {iter_time:.2f}s")
                store_items = copy.deepcopy(layer['items'])  # copy to iterate over
                to_remove, remove_head = mp.Array('i', len(store_items)), mp.Value('i')  # store indices so that cycle is not repeated if either item has already been removed / discarded
                for i in range(len(store_items)):
                    to_remove[i] = -1
                combinations_num = int(comb(N=len(store_items), k=2))
                input_queue = mp.JoinableQueue(num_workers * 2)
                output_queue = mp.Queue(500)  # no need to join
                # start processes
                processes = tuple(ItemMerger(i, closeness_thresh, dissimilarity_thresh, input_queue, output_queue,
                                        to_remove, remove_head, funcs).start() for i in range(num_workers))
                for (i, item0), (j, item1) in combinations(enumerate(store_items), 2):
                    if i == j or i in to_remove or j in to_remove:
                        continue  # items are the same or items have already been processed
                    try:
                        tile_rect0 = self.metadata[layer['name']]['tile_rect'][i]
                        tile_rect1 = self.metadata[layer['name']]['tile_rect'][j]
                    except KeyError as keyerr:
                        raise KeyError(f"{keyerr.args[0]} - Unspecified metadata for layer {layer['name']}")
                    except IndexError:
                        raise IndexError(f"Item {i if i > j else j} is missing from layer {layer['name']} metadata")
                    input_queue.put(((i, item0), (j, item1), (tile_rect0, tile_rect1)), timeout=timeout)
                    if num_put_to_process % 500 == 0:
                        logger.info(f"Items pairs: {num_put_to_process}/{combinations_num} queued - {sum(rm > -1 for rm in to_remove)} to merge")
                    num_put_to_process += 1
                for i in range(num_workers):
                    input_queue.put(None)  # put sentinel for processes to know when to stop

                input_queue.join()

                sentinel_count = 0
                while not output_queue.empty():  # will probably be empty when this point is reached !
                    data = output_queue.get(timeout=timeout)
                    if data is None:
                        sentinel_count += 1
                        if sentinel_count == num_workers:
                            break
                        else:
                            continue
                    (i, j), outer_points, (x_out, y_out, w_out, h_out) = data
                    self.add_item(layer['name'], items[i]['type'], class_=items[i]['class'], points=outer_points,
                                  tile_rect=(x_out, y_out, w_out, h_out))
                for item_idx in sorted(to_remove, reverse=True):  # remove items that were merged - must start from last index or will return error
                    # noinspection PyBroadException
                    try:
                        self.remove_item(layer['name'], item_idx)
                    except Exception:
                        logger.error(f"Failed to remove item {item_idx} in layer {layer['name']}", exc_info=True)
                changed = items != store_items
                merged_num += len(store_items) - len(items)
                num_iters += 1
                iter_time = iter_time + (time.time() - iter_start_time - iter_time) / num_iters
                logger.info("Wait for child processes joining main process ...")
                for process in processes:
                    process.join()
            if changed:
                logger.info(f"[Layer '{layer['name']}'] Max number of iterations reached ({num_iters})")
            else:
                logger.info(f"[Layer '{layer['name']}'] No changes after {num_iters} iterations.")
            layer_time = iter_time + (time.time() - layer_start_time - layer_time) / (r + 1)
            logger.info(f"[Layer '{layer['name']}'] {merged_num} total merged items in {layer_time:.2f}s.")
        self.merged = True
        logger.info("Done!")
        logger.removeHandler(fh)
        logger.removeHandler(ch)


# to be pickled, classes must be top level objects
Funcs = namedtuple('Funcs', ('euclidean_dist',
                             'check_relative_rect_positions',
                             'item_points',
                             'remove_overlapping_points',
                             'get_merged',
                             'find_closest_points'))


class ItemMerger(mp.Process):

    def __init__(self, id_, closeness_thresh, dissimilarity_thresh, input_queue, output_queue,
                 to_remove, remove_head, funcs):
        """
        Handles the merging of items, and returns the merged item to an output queue.
        :param id_:
        :param closeness_thresh:
        :param dissimilarity_thresh:
        :param input_queue:
        :param output_queue:
        :param to_remove:
        :param remove_head:
        :param funcs:
        """
        super().__init__(name='ItemMerger', daemon=True)
        self.id_ = id_
        self.closeness_thresh = closeness_thresh
        self.dissimilarity_thresh = dissimilarity_thresh
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.to_remove = to_remove
        self.remove_head = remove_head
        self.f = funcs  # namedtuple with all the required functions. Functions can be called with . operator.
        self.timeout = 10

    def run(self):
        logger = logging.getLogger(__name__)
        logger.info(f"Process {self.id_} started ...")
        merged_items = 0
        mean_merge_time = 0
        while True:
            try:
                data = self.input_queue.get(timeout=self.timeout)
            except queue.Empty:
                logger.error(f"{self.timeout} exceeded when waiting to get item from queue.")
                raise
            if data is None:
                self.input_queue.task_done()
                break
            (i, item0), (j, item1), (tile_rect0, tile_rect1) = data
            if i == j or i in self.to_remove or j in self.to_remove:  # __contains__ implemented in mp.Array
                self.input_queue.task_done()  # signal to queue that items have been processed
                continue  # items are the same or items have already been processed
            # Brute force approach, shapes are not matched perfectly, so if there are many points close to another
            # point and distance dosesn't match the points 1-to-1 paths might be merged even though they don't
            rects_positions, origin_rect, rect_areas = self.f.check_relative_rect_positions(tile_rect0, tile_rect1)
            if not rects_positions:
                self.input_queue.task_done()  # signal to queue that items have been processed
                continue  # do nothing if items are not from touching/overlapping tiles
            points_near, points_far = (tuple(self.f.item_points(item0 if origin_rect == 0 else item1)),
                                       tuple(self.f.item_points(item0 if origin_rect == 1 else item1)))
            if rects_positions == 'overlap':
                points_far = self.f.remove_overlapping_points(points_near, points_far)
                try:
                    points_far[0]
                except IndexError as err:
                    logging.error(str(err.args), exc_info=True)
            assert points_far, "Some points must remain from this operation, or positions should have been 'contained'"
            total_min_dist = 0.0
            # noinspection PyBroadException
            try:
                start_time = time.time()
                closest_points, point_dist = self.f.find_closest_points(self.f.euclidean_dist, points_near, points_far,
                                                                        self.closeness_thresh)
                if closest_points and len(closest_points) > 1:
                    total_min_dist += sum(
                        min(point_dist[p0][p1] if (p0, p1) in closest_points else 0 for p1 in points_far)
                        for p0 in points_near
                    )  # sum dist
                    if total_min_dist / len(closest_points) < self.dissimilarity_thresh:
                        # Signal to other processes ASAP that the items have been processed
                        if i not in self.to_remove and j not in self.to_remove:  # check that either items haven't been processed by other worker in the meantime
                            with self.remove_head.get_lock():  # needed for read + write (non-atomic) operation
                                self.to_remove[self.remove_head.value] = i
                                self.remove_head.value += 1
                            with self.remove_head.get_lock():  # needed for read + write (non-atomic) operation
                                self.to_remove[self.remove_head.value] = j
                                self.remove_head.value += 1
                        else:
                            self.input_queue.task_done()  # signal to queue that items have been processed
                            continue
                        outer_points = self.f.get_merged(points_near, points_far, closest_points)
                        # make bounding box for new contour
                        x_out, y_out = (min(tile_rect0[0], tile_rect1[0]), min(tile_rect0[1], tile_rect1[1]))
                        w_out = max(tile_rect0[0] + tile_rect0[2],
                                    tile_rect1[0] + tile_rect1[2]) - x_out  # max(x+w) - min(x)
                        h_out = max(tile_rect0[1] + tile_rect0[3],
                                    tile_rect1[1] + tile_rect1[3]) - y_out  # max(x+w) - min(x)
                        self.output_queue.put(((i, j), outer_points, (x_out, y_out, w_out, h_out)))
                        merged_items += 1
                        mean_merge_time = mean_merge_time + (time.time() - start_time - mean_merge_time) / merged_items  # running average
                        logger.info(f"Merged item {i} and {j} | mean merge time: {mean_merge_time:.2f}s total #merged: {merged_items}")

            except Exception:
                logger.error(f"""
                Items: {(i, j)} total item num:
                [Bounding box] 0 x: {tile_rect0[0]} y: {tile_rect0[1]} w: {tile_rect0[2]} h: {tile_rect0[2]}
                [Bounding box] 1 x: {tile_rect1[0]} y: {tile_rect1[1]} w: {tile_rect1[2]} h: {tile_rect1[2]}
                Result of rectangle position check: '{rects_positions}', origin: {origin_rect}, areas: {rect_areas}
                """)
                logger.error('Failed.', exc_info=True)
            self.input_queue.task_done()  # signal to queue that items have been processed
        self.output_queue.put(None)  # sentinel for breaking out of queue-get loop
        self.output_queue.close()
        logger.info(f"Terminating process {self.id_} | mean merge time: {mean_merge_time:.2f}s total #merged: {merged_items}")


def main():
    r"""Load existing annotation and merge contours"""
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--annotation_path', required=True, type=Path)
    parser.add_argument('--closeness_threshold', type=float, default=5.0, help="")
    parser.add_argument('--dissimilarity_threshold', type=float, default=4.0, help="")
    parser.add_argument('--max_iter', type=int, default=1, help="")
    parser.add_argument('--num_workers', type=int, default=4, help="")
    parser.add_argument('--log_dir', type=str, default='', help="")
    parser.add_argument('--append_merged_suffix', action='store_true', help="")
    parser.add_argument('--sequential', action='store_true')
    args, unparsed = parser.parse_known_args()
    with open(args.annotation_path, 'r') as annotation_file:
        annotation_json = json.load(annotation_file)
    builder = AnnotationBuilder.from_object(annotation_json)
    builder.merge_overlapping_segments(closeness_thresh=5.0, dissimilarity_thresh=4.0, max_iter=1,
                                       parallel=not args.sequential, num_workers=4, log_dir='')
    builder.dump_to_json(args.annotation_path.parent, name='merged' if args.append_merged_suffix else '')


if __name__ == '__main__':
    main()

# @staticmethod
# def find_extremes(close_point_pairs, distance):
#     """
#     Finds extreme points based on close point pairs between two shapes
#     :param close_point_pairs:
#     :param distance:
#     :return:
#     """
#     close_point_pairs = tuple(close_point_pairs)
#     # find vector representation of line
#     points = tuple(p for p in chain.from_iterable(close_point_pairs))
#     x = np.array(tuple([1, p[0]] for p in points))  # first basis function is for bias, second is linear (slope)
#     y = np.array(tuple(p[1] for p in points))
#     weights = np.linalg.pinv(x.T @ x) @ x.T @ y
#     bias, slope = weights
#     offset = np.array([0, slope*0 + bias])
#     direction = np.array([1, slope*1 + bias])
#     def find_projection(v, r, s):
#         return r + s * np.dot(v, s) / np.dot(s, s)
#     projection_pairs = [(find_projection(p0, offset, direction), find_projection(p1, offset, direction))
#                         for p0, p1 in close_point_pairs]
#     centre_of_mass = np.mean([p for p in chain.from_iterable(projection_pairs)])  # points are aligned
#     extremes = [
#         close_point_pairs[
#             np.argmin([min(distance(p0, centre_of_mass), distance(p1, centre_of_mass))
#                        for p0, p1 in projection_pairs])
#         ],  # highest points
#         close_point_pairs[
#             np.argmax([max(distance(p0, centre_of_mass), distance(p1, centre_of_mass))
#                        for p0, p1 in projection_pairs])
#         ]  # lowest points
#     ]
#     topmost_first_extremes = sorted(extremes, key=lambda point_pair: point_pair[0][1])  # y-axis of first point
#     return topmost_first_extremes





#
# def get_merged(points_near, points_far, close_point_pairs, positions='horizontal'):
#     """
#     Function to merge contours from adjacent tiles
#     :param points_near: leftmost (for positions = horizontal) or topmost (for position = vertical) path
#     :param points_far:
#     :param close_point_pairs:
#     :param positions
#     :return: merged points
#     """
#     close_point_pairs = tuple(close_point_pairs)
#     # two-way mapping
#     correspondance = {p0: p1 for p0, p1 in close_point_pairs}
#     correspondance.update({p1: p0 for p0, p1 in close_point_pairs})
#     # drop close segments
#     outer_points = []
#     if positions == '':
#         raise ValueError("Cannot merge items if bounding boxes are not at least adjacent")
#     if positions == 'overlap':
#         raise NotImplementedError(f"No need to merge overlapping contours")
#     # https://stackoverflow.com/questions/45323590/do-contours-returned-by-cvfindcontours-have-a-consistent-orientation
#     # outer contours are oriented counter-clockwise
#     # scanning for first point is done from top left to bottom right
#     # assuming contours are extracted for each value independently
#     if positions == 'horizontal':
#         extremes = [
#             close_point_pairs[
#                 np.argmin([min(p0[1], p1[1]) for p0, p1 in close_point_pairs])
#             ],  # highest points
#             close_point_pairs[
#                 np.argmax([max(p0[1], p1[1]) for p0, p1 in close_point_pairs])
#             ]  # lowest points
#         ]
#         for i0, point0 in enumerate(points_near):
#             # start - lower extreme - higher - extreme
#             outer_points.append(point0)
#             if point0 == extremes[1][0]:  # p0 of lowest (p0, p1) pair
#                 break
#         start_p1 = correspondance[point0]
#         start_p1_idx = points_far.index(start_p1)
#         for i1, point1 in enumerate(points_far[start_p1_idx:] + points_far[:start_p1_idx]):
#             outer_points.append(point1)
#             if point1 == extremes[0][1]:  # p1 of highest (p0, p1) pair
#                 break
#         restart_p0 = correspondance[point1]
#         for i00, point0 in enumerate(points_near[points_near.index(restart_p0):]):
#             outer_points.append(point0)
#             i0 += 1
#         assert outer_points
#     elif positions == 'vertical' or positions == 'overlap':
#         extremes = [
#             close_point_pairs[
#                 np.argmin([min(p0[0], p1[0]) for p0, p1 in close_point_pairs])
#             ],  # leftmost points
#             close_point_pairs[
#                 np.argmax([max(p0[0], p1[0]) for p0, p1 in close_point_pairs])
#             ]  # rightmost points
#         ]
#         for i0, point0 in enumerate(points_near):
#             # start - lower extreme - higher - extreme
#             outer_points.append(point0)
#             if point0 == extremes[0][0]:  # p0 of rightmost (p0, p1) pair
#                 break
#             assert i0 < extremes[0] and i0 < extremes[1]
#         start_p1 = correspondance[point0]
#         start_p1_idx = points_far.index(start_p1)
#         for i1, point1 in enumerate(points_far[start_p1_idx:] + points_far[:start_p1_idx]):
#             outer_points.append(point1)
#             if point1 == extremes[1][1]:  # p1 of rightmost (p0, p1) pair
#                 break
#         restart_p0 = correspondance[point1]
#         for i00, point0 in enumerate(points_near[points_near.index(restart_p0):]):
#             outer_points.append(point0)
#             i0 += 1
#         assert outer_points
#     return outer_points