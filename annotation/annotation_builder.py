import json
import math
import copy
import warnings
from collections import defaultdict, Counter
from itertools import tee
from pathlib import Path
import multiprocessing as mp
import numpy as np
from matplotlib import pyplot as plt
import cv2


class AnnotationBuilder:
    r"""
    Use to create AIDA annotation .json files for a single images/WSI that can be read by AIDA
    """
    @classmethod
    def from_annotation_path(cls, path):
        path = Path(path)
        with open(path, 'r') as file:
            obj = json.load(file)
        if 'annotation' in obj:
            obj = obj['annotation']  # for annotations exported from Nasullah's backend
        if 'slide_id' not in obj:
            obj['slide_id'] = path.with_suffix('').name
        if 'project_name' not in obj:
            obj['project_name'] = path.with_suffix('').name
        return cls.from_object(obj)

    @classmethod
    def from_object(cls, obj, metadata_warning=False):
        if 'annotation' in obj:
            obj = obj['annotation']  # for annotations exported from Nasullah's backend
        instance = cls(
            obj['slide_id'] if 'slide_id' in obj else 'unknown_slide',
            obj['project_name'] if 'project_name' in obj else '',
            obj['layer_names'] if 'layer_names' in obj else [layer['name'] for layer in obj['layers']]
        )
        instance._obj = obj
        try:
            instance.metadata = obj['metadata']
        except KeyError:
            if metadata_warning:
                warnings.warn(f"No available metadata for {instance.slide_id}")
        return instance

    def __init__(self, slide_id, project_name='', layers=()):
        self.slide_id = slide_id  # AIDA uses filename before extension to determine which annotation to load
        self.project_name = project_name if project_name else slide_id
        self._obj = {
            'name': self.project_name,
            'layers': [],
            'data': {}  # additional data, e.g. network that produced annotation
        }  # to be dumped to .json
        self.layer_names = []
        # store additional info on segments for processing
        self.last_added_item = None
        # point comparison
        for layer_name in layers:
            self.add_layer(layer_name)  # this also updates self.layer_names with references to layers names

    @classmethod
    def concatenate(cls, annotation0, annotation1, concatenate_layers=True):
        r"""Add layers from another annotation object to this object"""
        annotation0 = copy.copy(annotation0)
        for layer in annotation1._obj['layers']:
            if layer['name'] in annotation0.layer_names and concatenate_layers:
                layer0 = next(layer_ for layer_ in  annotation0._obj['layers'] if layer_['name'] == layer['name'])
                layer0['items'].extend(layer['items'])
            else:
                if layer['name'] in annotation0.layer_names:
                    layer['name'] = layer['name'] + ('_' + annotation0.project_name)
                annotation0._obj['layers'].append(layer)
                annotation0.layer_names.append(layer['name'])
        return annotation0

    @staticmethod
    def euclidean_dist(p1, p2):
        x1, y1 = (p1['x'], p1['y']) if isinstance(p1, dict) else p1
        x2, y2 = (p2['x'], p2['y']) if isinstance(p2, dict) else p2
        return math.sqrt(math.pow(x2 - x1, 2) + math.pow(y2 - y1, 2))

    def __getitem__(self, idx):
        if isinstance(idx, str):
            idx = self.get_layer_idx(idx)
        return self._obj['layers'][idx]

    def __setitem__(self, idx, value):
        self._obj['layers'][idx] = value

    def num_layers(self):
        return len(self._obj['layers'])

    def __len__(self):
        return 0 if not self.num_layers() else sum(len(layer['items']) for layer in self._obj['layers'])

    def rename_layer(self, old_name, new_name):
        layer_idx = self.get_layer_idx(old_name) if type(old_name) is str else old_name  # if name is given rather than index
        layer = self._obj['layers'][layer_idx]
        layer['name'] = new_name
        self.layer_names[layer_idx] = new_name

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
        self.layer_names.append(self._obj['layers'][-1]['name'])
        return self

    def add_item(self, layer_idx, type_, class_=None, points=None, filled=False, color=None, rectangle=None):
        r"""
        Add item to desired layer
        :param layer_idx: numerical index or layer name (converted to index)
        :param type_:
        :param class_:
        :param points:
        :param tile_rect:
        :param filled:
        :param color:
        :return:
        """
        if type(layer_idx) is str:
            layer_idx = self.get_layer_idx(layer_idx)  # if name is given rather than index
        if color is None:
            color = {} if not filled else {
                "fill": {
                    "saturation": 0.44,
                    "lightness": 0.69,
                    "alpha": 0.7,
                    "hue": 170
                },
                "stroke": {
                    "saturation": 0.44,
                    "lightness": 0.69,
                    "alpha": 1,
                    "hue": 170
                }  # fill with pre-selected color
            }
        new_item = {
            'class': class_ if class_ else self._obj['layers'][layer_idx]['name'],
            'type': type_,
            'closed': True,
            'color': color
        }
        if type_ == 'path':
            new_item.update(segments=[])
            self._obj['layers'][layer_idx]['items'].append(new_item)
            if points:
                self.add_segments_to_last_item(points)
        elif type_ == 'rectangle':
            if rectangle is not None:
                if isinstance(rectangle, (tuple, list)):
                    new_item.update(x=rectangle[0], y=rectangle[1], width=rectangle[2], height=rectangle[3])
                elif isinstance(rectangle, dict) and set(rectangle.keys()) == {'x', 'y', 'width', 'height'}:
                    new_item.update(**rectangle)
                else:
                    raise ValueError(f"Unsupported format for rectangle {rectangle}")
            self._obj['layers'][layer_idx]['items'].append(new_item)
        self.last_added_item = self._obj['layers'][layer_idx]['items'][-1]  # use to update item
        return self

    def remove_item(self, layer_idx, item_idx):
        layer_idx = self.get_layer_idx(layer_idx)  # FIXME slightly confusing - this is getting layer idx from name or doing nothing
        del self._obj['layers'][layer_idx]['items'][item_idx]

    def set_layer_items(self, layer_idx, items):
        self._obj['layers'][layer_idx]['items'] = items
        return self

    def add_segments_to_last_item(self, points, version='new'):
        segments = []
        for i, point in enumerate(points):
            x, y = point
            if version == 'old':
                new_segment = {
                    'point': {
                        'x': x,
                        'y': y,
                    }
                  }
            elif version == 'new':
                new_segment = [
                    [x, y]
                ]
            else:
                raise ValueError(f"version must be 'new' or 'old' (not {version})")
            segments.append(new_segment)
        self.last_added_item['segments'] = segments
        return self

    def print(self, indent=4):
        print(json.dumps(self._obj, sort_keys=False, indent=indent))

    def export(self):
        r"""
        Add attributes so that obj can be used to create new data annotation
        :return:
        """
        obj = copy.deepcopy(self._obj)
        obj['project_name'] = self.project_name
        obj['slide_id'] = self.slide_id
        obj['layer_names'] = self.layer_names
        return obj

    def dump_to_json(self, save_dir, name='', suffix_to_remove=('.ndpi', '.svs', '.json'), rewrite_name=False):  # adding .json here so that .json is not written twice
        for layer_name, count in Counter(tuple(layer['name'] for layer in self._obj['layers'])).items():
            if count > 1:
                raise ValueError(f"Duplicate layer {layer_name} in annotation {self.slide_id}")
        if name and rewrite_name:
            save_path = Path(save_dir)/name
        else:
            save_path = Path(save_dir)/self.slide_id
        save_path = save_path.with_suffix('.json') if save_path.suffix in suffix_to_remove else \
            save_path.parent/(save_path.name +'.json')  # add json taking care of file ending in .some_text.[ext,no_ext]
        if name and not rewrite_name:
            save_path = str(save_path)[:-5] + '_' + name + '.json'
        with open(save_path, 'w') as dump_file:
            json.dump(self.export(), dump_file)

    def get_layer_points(self, layer_idx, contour_format=True):
        r"""Get all paths in a given layer, function used to extract layer from annotation object"""
        # TODO add code in item_points to convert circles into contours
        if isinstance(layer_idx, str):
            layer_idx = self.get_layer_idx(layer_idx)
        layer = self._obj['layers'][layer_idx]
        if contour_format:
            layer_points = list(
                np.array(list(self.item_points(item))).astype(np.int32)[:, np.newaxis, :]  # contour functions only work on int32
                if (item['type'] == 'path' and 'segments' in item and item['segments']) or
                   (item['type'] == 'rectangle')
                else np.array([])
                for item in layer['items']
            )
        else:
            layer_points = list(list(self.item_points(item)) for item in layer['items'])
        return layer_points, layer['name']

    def get_circles(self):
        r"""Extracts all circle annotations in center-radius format"""
        circles = dict()
        for layer in self._obj['layers']:
            layer_circles = circles[layer['name']] = []
            for item in layer['items']:
                if item['type'] != 'circle':
                    continue
                layer_circles.append({
                    'center': (item['center']['x'], item['center']['y']),
                    'radius': item['radius']
                })
        return circles

    def shrink_paths(self, factor=0.5, min_point_density=0.1, min_point_num=10):
        r"""Function to remove points from contours in order to decrease the annotation size
        :param: factor: factor by which the contours will be shrunk
        :param: min_point_density: minimum number of points per pixel. Contours with less points will not be shrunk.
        E.g. 0.1 means one point every ten pixels
        """
        new_layers = []
        for layer in self._obj['layers']:
            new_layer = copy.deepcopy(layer)
            new_layer['items'].clear()
            for i, item in enumerate(layer['items']):
                if len(item['segments']) > 1:
                    new_item = copy.deepcopy(item)
                    item_contour = np.array(list(self.item_points(item))).astype(np.int32)[:, np.newaxis, :]  # contour functions only work on int32
                    x, y, w, h = cv2.boundingRect(item_contour)
                    if not (len(new_item['segments']) / 4) / np.sqrt(w*h) < min_point_density \
                            and len(item['segments']) > min_point_num:
                        # remove points with minimal distance between them till desired shrinkage is attained
                        new_len = round(len(item['segments']) * (1 - factor))  # length of new element after point removal
                        # distances from smallest to greatest, with index of first point along contour
                        indexed_distances = sorted(((j, self.euclidean_dist(p1, p2))
                                                    for j, (p1, p2) in enumerate(pairwise(self.item_points(item)))),
                                                   key=lambda indexed_el: indexed_el[1])
                        # reorder so that first eliminated point is last in the list,
                        # so that index of next point is preserved
                        indexed_distances = sorted(indexed_distances[:(len(new_item['segments']) - new_len)],
                                                   key=lambda indexed_el: indexed_el[0], reverse=True)

                        for idx, dist in indexed_distances:
                            del new_item['segments'][idx]
                        assert len(new_item['segments']) == new_len, "Point remsum(1 for l in lengths.values() if l)oval must yield item of expected length"
                    new_layer['items'].append(new_item)
            new_layers.append(new_layer)
        self._obj['layers'] = new_layers

    def split(self, mode: str, n: int, min_len=5, save_dir=None, roi_layer=None) -> tuple:
        r"""
        Split annotation into multiple dictionary files
        :param mode: 'parts' or 'size'. Determines how to interpret parameter n
        :param n: mode == 'parts': number of dicts to split annotation into;
                  mode == 'size': split annotation so that each part has max size n
        :param min_len: if number of items is less than min_len
        :param save_dir: if not None, splits are saved into json files in this directory
        :param roi_layer: contours inside the contours in this layers are split in a spatially equal manner with mode=spatial
        :return: annotation splits
        """
        if mode == 'parts' or mode == 'spatial':
            num_splits = n
        elif mode == 'size':
            num_splits = max(len(layer['items']) for layer in self._obj['layers'])
        else:
            raise ValueError(f"Illegal mode '{mode}'. Mode must be in ['parts', 'size', 'spatial']")
        splits = tuple(
            {
                'name': self.project_name,
                'layers': list({
                                   'name': layer['name'],
                                   'opacity': 1,
                                   'items': []
                               } for layer in self._obj['layers']),
                'slide_id': self.slide_id,
                'project_name': self.project_name
            } for m in range(num_splits)
        )
        if mode in ('parts', 'size'):
            for i, layer in enumerate(self._obj['layers']):
                if len(layer['items']) < min_len:
                    chunks = (layer['items'],) * num_splits  # repeat layer in each chunk if layer is very small
                else:
                    if mode == 'parts':
                        chunks = tuple(list(chunk) for chunk in np.array_split(layer['items'], num_splits))
                    elif mode == 'size':
                        chunk_len = round(len(layer['items']) / num_splits)
                        chunks = list(layer['items'][i:i + chunk_len] for i in range(0, len(layer['items']), chunk_len))
                for j, part in enumerate(splits):
                    part['layers'][i]['items'] = chunks[j]
        else:
            if roi_layer not in self.layer_names:
                raise ValueError(f"For 'spatial' mode, roi_layer must be name of top level layer (no layer named '{roi_layer}')")
            roi_contours, layer_name = self.get_layer_points(roi_layer, contour_format=True)
            roi_contour = max((contour for contour in roi_contours if contour.size > 0), key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(roi_contour)
            if w > h:
                split_rects = [(x + i*int(w/num_splits), y, int(w/num_splits), h) for i in range(num_splits)]
            else:
                split_rects = [(x, y + i*int(h/num_splits), w, int(h/num_splits)) for i in range(num_splits)]
            for i, layer in enumerate(self._obj['layers']):
                chunks = [[] for i in range(num_splits)]
                layer_contours, _ = self.get_layer_points(layer['name'], contour_format=True)
                for j, contour in enumerate(layer_contours):
                    for k, split_rect in enumerate(split_rects):
                        bounding_rect = cv2.boundingRect(contour)
                        overlap, origin, areas = AnnotationBuilder.check_relative_rect_positions(
                            split_rect,
                            bounding_rect)
                        if overlap == 'overlap' or overlap == 'contained' and origin == 0:
                            chunks[k].append(layer['items'][j])
                            break  # do not insert a contour in two different layers !
                for j, part in enumerate(splits):
                    part['layers'][i]['items'] = chunks[j]
        if save_dir is not None:
            splits_dir = save_dir/(self.slide_id + '_splits')
            splits_dir.mkdir(exist_ok=True)
            for m, split in enumerate(splits):
                split_annotation = AnnotationBuilder.from_object(split)
                split_annotation.dump_to_json(splits_dir, name=str(m))
        return splits

    @classmethod
    def merge(cls, annotations_dir, slide_id):  # TODO test
        annotations_dir = Path(annotations_dir)
        split_paths = [path for path in annotations_dir.iterdir() if slide_id in path.name and path.name.endswith('.json')]
        if not split_paths:
            raise FileNotFoundError(f"No matching files for slide id: '{slide_id}'")
        splits = [cls.from_annotation_path(split_path) for split_path in split_paths]
        split = splits[0]
        split.slide_id = slide_id
        for i in range(1, len(splits)):
            split = cls.concatenate(split, splits[i])
        return split

    def summary_plot(self, bins=8):
        lengths = dict((layer_name, []) for layer_name in self._obj['layer_names'])
        areas = dict((layer_name, []) for layer_name in self._obj['layer_names'])
        for layer in self._obj['layers']:
            layer_lengths = lengths[layer['name']]
            layer_areas = areas[layer['name']]
            for i, item in enumerate(layer['items']):
                layer_lengths.append(len(item['segments']))
                item_contour = np.array(list(self.item_points(item))).astype(np.int32)[:, np.newaxis, :]
                layer_areas.append(cv2.contourArea(item_contour))
        fig, axes = plt.subplots(2, sum(1 for l in lengths.values() if l))
        message = ""
        for i, layer in enumerate(self._obj['layers']):
            if not lengths[layer['name']]:
                i -= 1
                continue
            layer_lengths = lengths[layer['name']]
            length_mean = np.mean(layer_lengths)
            length_std = np.std(layer_lengths)
            axes[0, i].hist(lengths[layer['name']], bins=bins)
            layer_length_message = f"{layer['name']} num points: m{round(length_mean)}, s{round(length_std)}"
            message += (' ' + layer_length_message)
            axes[0, i].set_title(layer_length_message)
            layer_areas = areas[layer['name']]
            area_mean = np.mean(layer_areas)
            area_std = np.std(layer_areas)
            axes[1, i].hist(areas[layer['name']], bins=bins)
            layer_area_message = f"{layer['name']} contour area: m{round(area_mean)}, s{round(area_std)}"
            message += (' ' + layer_area_message + '\n')
            axes[0, i].set_title(layer_area_message)
        print(message)
        return message

    @staticmethod
    def item_points(item):
        """Generator over item's points"""
        # remove multiple identical points - as they can cause errors
        if item['type'] == 'path':
            points = set()
            for segment in item['segments']:
                if 'point' in segment:  # point = {x: v, y: v} format
                    point = tuple(segment['point'].values())
                elif isinstance(segment[0], (list, tuple)):  # couple format
                    point = tuple(segment[0])  # in case graphic handles are present
                else:
                    point = tuple(segment)  # in case only point is present
                if point not in points:
                    points.add(point)
                    yield point
        elif item['type'] == 'rectangle':
            if 'from' in item and 'to' in item:
                upper_left_corner = tuple(item['from'].values())  # x and y
                bottom_left_corner = tuple(item['to'].values())  # x and y
                x, y = upper_left_corner
                w, h = bottom_left_corner[0] - x, bottom_left_corner[1] - y
            elif 'x' in item and 'y' in item and 'width' in item and 'height' in item:
                x, y, w, h = item['x'], item['y'], item['width'], item['height']
            else:
                raise ValueError(f"Invalid rectangle layer item \n{item}\n Either 'from' and 'to' or 'x', 'y', 'width', and 'height' must be specified")
            yield (x, y)  # yield points anticlockwise
            yield (x, y + h)
            yield (x + w, y + h)
            yield (x + w, y)
        else:
            raise NotImplementedError(f"Unrecognized item type '{item['type']}'")

    def filter(self, layer_idx, functions, contour_format=True, workers=4, **kwargs):
        if isinstance(layer_idx, str):
            layer_idx = self.get_layer_idx(layer_idx)
        layers = self._obj['layers']
        layer = layers[layer_idx]
        num_items = len(layer['items'])

        def check_item(item):
            if contour_format:
                points = np.array(list(self.item_points(item))).astype(np.int32)[:, np.newaxis, :]
            else:
                points = self.item_points(item)
            if any(function(points, **kwargs) for function in functions):
                return True
        with mp.Pool(workers) as pool:
            to_keep = pool.starmap(check_item, layer['items'])
        for index, keep_flag in enumerate(to_keep):
            if keep_flag:
                del layer['items'][index]
        new_num_items = len(layer['items'])
        print(f"{num_items - new_num_items} were deleted from layer '{layer_idx}'")

    @staticmethod
    def check_relative_rect_positions(tile_rect0, tile_rect1, eps=0):
        r"""
        :param tile_rect0:
        :param tile_rect1:
        :param eps: tolerance in checks (not in contained check !)
        :return: positions: overlap|horizontal|vertical|'' - the relative location of the two paths
                 origin_rect: meaning depends on relative positions of two boxes:
                            * contained: which box is bigger
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
        y_contained = (y0 <= y1 <= y_h0 and y0 <= y_h1 <= y_h0) or \
                      (y1 <= y0 <= y_h1 + eps and y1 - eps <= y_h0 <= y_h1)
        if x_contained and y_contained:
            positions = 'contained'
            if (x_w0 - x0) * (y_h0 - y0) >= (x_w1 - x1) * (y_h1 - y1):
                origin_rect = 0  # which box is bigger
            else:
                origin_rect = 1
        elif not x_contained and y_overlap and (x_w0 <= x1 + eps or x_w1 <= x0 + eps):
            positions = 'horizontal'
            if x_w0 <= x1 + eps:
                origin_rect = 0
            elif x_w1 <= x0 + eps:
                origin_rect = 1
            else:
                raise ValueError("shouldn't be here")
        elif not y_contained and x_overlap and (y_h0 <= y1 + eps or y_h1 <= y0 + eps):
            positions = 'vertical'
            if y_h0 <= y1 + eps:
                origin_rect = 0
            elif y_h1 <= y0 + eps:
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
        return positions, origin_rect, rect_areas

    def add_data(self, key, value):
        self._obj['data'][key] = value

    def old_to_new(self):
        r"""
        Translates annotation in old annotation schema to newer annotation schema
        This corresponds to a bunch of changes, which should be routinely updated
        16/12/2019
        Rectangle: was specified by 'from' and 'to' attributes. Is now specified by 'x', 'y', 'width' and 'length
        """
        num_changed_items = 0
        for layer in self._obj['layers']:
            for item in layer['items']:
                if item['type'] == 'rectangle' and 'from' in item and 'to' in item:
                    x, y = tuple(item['from'].values())
                    xw, yh = tuple(item['to'].values())
                    w, h = xw - x, yh - y
                    item['x'], item['y'], item['width'], item['height'] = x, y, w, h
                    # keep from and to attributes for backward compatibility
                    num_changed_items += 1
        print(f"{num_changed_items} items were updated")


def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


