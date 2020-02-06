import sys
import json
import math
import copy
import warnings
from collections import defaultdict, namedtuple
from itertools import tee
import logging
from datetime import datetime
import time
import multiprocessing as mp
from pathlib import Path
import numpy as np
from scipy.spatial.distance import cdist
from matplotlib import pyplot as plt
import cv2
import queue
import tqdm
import psutil
from .annotation_builder import AnnotationBuilder


class AnnotationPathMerger(AnnotationBuilder):

    def __init__(self, slide_id, project_name, layers=(), keep_original_paths=False):
        super().__init__(slide_id, project_name, layers, keep_original_paths)

    @staticmethod
    def find_extreme_points(points_near, points_far, positions, closeness_thresh=300.0):
        r"""Finds pairs of closest points in contours, and returns them if they are within a distance threshold
        :param distance: distance metric taking two points and returning commutative value
        :param points_near:
        :param points_far:
        :param rect_near
        :param rect_far
        :param positions: relative position of two bounding contours - supports 'horizontal' or 'vertical'
        :param closeness_thresh: decide whether closest pair of points between contour is close enough for the contours
        to be considered adjacent
        :return: pairs of corresponding points
                 distance between point pairs
        """
        # find closest points
        assert positions in ['horizontal', 'vertical']
        dist_mat = cdist(np.array(points_near), np.array(points_far), 'euclidean')
        closest_points_n, closest_points_f = [], []
        for n in range(len(points_near)):
            f = np.argmin(dist_mat[n, :])
            if dist_mat[n, f] < closeness_thresh:
                closest_points_n.append(points_near[n])
                closest_points_f.append(points_far[f])
                # make sure far points are not selected twice -- need to capture full extent of contour at border
                dist_mat[:, f] = 10000.0
        closest_points_n = np.array(closest_points_n)
        closest_points_f = np.array(closest_points_f)
        # find extreme points
        if closest_points_n.size > 0 and closest_points_f.size > 0:
            # top / near
            top_near_points = closest_points_n[closest_points_n[:, 1] == closest_points_n[:, 1].min(), :]  # min y
            top_leftmost_near_point = top_near_points[np.argmin(top_near_points[:, 0])]  # min x
            top_rightmost_near_point = top_near_points[np.argmax(top_near_points[:, 0])]  # max x
            # bottom / near
            bottom_near_points = closest_points_n[closest_points_n[:, 1] == closest_points_n[:, 1].max(), :]  # max y
            bottom_leftmost_near_point = bottom_near_points[np.argmin(bottom_near_points[:, 0])]  # min x
            bottom_rightmost_near_point = bottom_near_points[np.argmax(bottom_near_points[:, 0])]  # max x
            # top / far
            top_far_points = closest_points_f[closest_points_f[:, 1] == closest_points_f[:, 1].min(), :]
            top_leftmost_far_point = top_far_points[np.argmin(top_far_points[:, 0])]
            top_rightmost_far_point = top_far_points[np.argmax(top_far_points[:, 0])]
            # bottom / far
            bottom_far_points = closest_points_f[closest_points_f[:, 1] == closest_points_f[:, 1].max(), :]
            bottom_leftmost_far_point = bottom_far_points[np.argmin(bottom_far_points[:, 0])]
            bottom_rightmost_far_point = bottom_far_points[np.argmax(bottom_far_points[:, 0])]

            if positions == 'horizontal':
                extreme_points = {
                    'pns': tuple(top_rightmost_near_point),  # TODO does it need to be leftmost (as in findContours search or can it be rightmost ? (closer to boundary)
                    'pne': tuple(bottom_rightmost_near_point),
                    'pfs': tuple(bottom_leftmost_far_point),
                    'pfe': tuple(top_leftmost_far_point)
                }
            elif positions == 'vertical':
                extreme_points = {
                    'pns': tuple(bottom_rightmost_near_point),
                    'pne': tuple(bottom_leftmost_near_point),
                    'pfs': tuple(top_leftmost_far_point),
                    'pfe': tuple(top_rightmost_far_point)
                }
            else:
                raise NotImplementedError("Overlapping contours must be made into vertical / horizontal by removing points !")
        else:
            extreme_points = None
        return extreme_points

    @staticmethod
    def get_merged(points_near, points_far, close_points):
        """
        Function to merge contours from adjacent tiles
        :param points_near: leftmost (for positions = horizontal) or topmost (for position = vertical) path
        :param points_far:
        :param close_points:
        :param positions
        :return: merged points
        """
        # drop close segments
        outer_points, point_memberships = [], []
        # https://stackoverflow.com/questions/45323590/do-contours-returned-by-cvfindcontours-have-a-consistent-orientation
        # outer contours are oriented counter-clockwise
        # scanning for first point is done from top left to bottom right
        # assuming contours are extracted for each value independently
        near_start_idx = points_near.index(close_points['pns'])
        near_end_idx = points_near.index(close_points['pne'])
        if near_start_idx <= near_end_idx:
            points_near_reordered = points_near[near_start_idx:near_end_idx]  # could be empty
        else:
            points_near_reordered = points_near[near_start_idx:] + points_near[:near_end_idx]
        for i0, point0 in enumerate(points_near_reordered):
            # start - lower extreme - higher - extreme
            outer_points.append(point0)
            point_memberships.append(0)
        far_start_idx = points_far.index(close_points['pfs'])
        far_end_idx = points_far.index(close_points['pfe'])
        if far_start_idx <= far_end_idx:
            points_far_reordered = points_far[far_start_idx:far_end_idx]  # c   ould be empty
        else:
            points_far_reordered = points_far[far_start_idx:] + points_far[:far_end_idx]
        for i1, point1 in enumerate(points_far_reordered):
            outer_points.append(point1)
            point_memberships.append(1)
        return outer_points

    @staticmethod
    def get_merged_many_close_points(points_near, points_far, close_point_pairs, positions='horizontal'):
        # FIXME stopping at wrong point
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
        for i1, point1 in enumerate(points_far[start_p1_idx:] + points_far[:start_p1_idx+1]):
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

    def get_paths_centroids_and_radii(self, layer_name):
        r"""Compute the centroid for all the paths in one layer"""
        layer_contours, layer_name = self.get_layer_points(layer_name, contour_format=True)
        centroids, radii = [], []
        for contour in layer_contours:
            if contour.size > 0:
                contour_moments = cv2.moments(contour)
                centroids.append((
                    contour_moments['m10'] / contour_moments['m00'],
                    contour_moments['m01'] / contour_moments['m00']
                ))
                radii.append(np.sqrt(cv2.contourArea(contour) / np.pi))
            else:
                centroids.append((0, 0))
                radii.append(0.0)
        return centroids, radii

    def merge_overlapping_segments(self, centroid_thresh=300.0, closeness_thresh=500.0, max_iter=1,
                                   parallel=False, num_workers=4, log_dir='', timeout=60, size_threshold=0.1):
        """
        Compares all segments and merges overlapping ones
        NB: must try different thresholds for closeness points
        :param: closeness_thresh: upper bound threshold for pair of points to be considered as belonging to adjacent contours
        :param: max_iter: number of merging iterations.
        """
        # TODO threshold by size percentile
        if parallel:
            # FIXME parallel implementation not up to date
            raise NotImplementedError("parallel implementation not up to date")
            self.parallel_merge_overlapping_segments(closeness_thresh, max_iter, num_workers,
                                                     log_dir=log_dir, timeout=timeout, centroid_thresh=centroid_thresh)
        else:
            # Set up logging
            logger = logging.getLogger(__name__)
            logger.setLevel(logging.DEBUG)
            fh = logging.FileHandler(Path(log_dir if log_dir else '.') / f'merge_segments_{datetime.now()}.log')  # logging to file for debugging
            fh.setLevel(logging.DEBUG)
            ch = logging.StreamHandler()  # logging to console for general runtime info
            ch.setLevel(logging.DEBUG)  # if this is set to ERROR, then errors are not printed to log, and vice versa
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            fh.setFormatter(formatter)
            ch.setFormatter(formatter)
            logger.addHandler(fh)
            logger.addHandler(ch)
            logger.info("Begin segment merging ...")
            for layer in self._obj['layers']:
                # no need to handle negative results
                changed = True
                num_iters = 0
                items = layer['items']
                logger.info(f"[Layer '{layer['name']}'] Initial number of items is {len(items)}. Thresholds: centroid: {centroid_thresh}, closeness: {closeness_thresh}")
                merged_num = 0  # keep count of change in elements number
                while changed and num_iters < max_iter and len(items) > 0:
                    paths_centroids, paths_radii = self.get_paths_centroids_and_radii(layer['name'])
                    if len(paths_centroids) < 2:
                        continue
                    centroids = np.array(paths_centroids)
                    dist_matrix = cdist(centroids, centroids, 'euclidean')
                    # subtract radii of contours from distance matrix
                    radial_matrix_i, radial_matrix_j = np.meshgrid(paths_radii, paths_radii)
                    radial_pair_sum_matrix = radial_matrix_i + radial_matrix_j
                    radial_pair_sum_matrix -= radial_pair_sum_matrix.mean()  # remove mean so that bigger glands have higher chance of being merged
                    dist_matrix -= radial_pair_sum_matrix  # so that threshold is between border points (approximately)
                    # repeat with newly formed contours
                    items = layer['items']
                    if num_iters == 0:
                        logger.info(f"[Layer '{layer['name']}'] First iteration ...")
                    else:
                        logger.info(f"[Layer '{layer['name']}'] Iter #{num_iters} cycle merged {len(store_items) - len(items)}; Items in layer: {len(items)}")
                    store_items = copy.deepcopy(layer['items'])  # copy to iterate over
                    to_remove = set()  # store indices so that cycle is not repeated if either item has already been removed / discarded
                    avg_num_pairs_per_contour = 0.0
                    for i, item0 in enumerate(tqdm.tqdm(store_items)):  # len(store_items) choose 2
                        # get near contours
                        close_paths_idx = np.where(dist_matrix[i, :] < centroid_thresh)[0]
                        close_paths = tuple(store_items[idx] for idx in close_paths_idx)
                        avg_num_pairs_per_contour = avg_num_pairs_per_contour + (
                                len(close_paths) - avg_num_pairs_per_contour) / (i + 1)
                        combinations_num = avg_num_pairs_per_contour * len(store_items)
                        for j, item1 in enumerate(close_paths):
                            idx_j = close_paths_idx[j]  # get position of contour inside array
                            if i == idx_j or i in to_remove or idx_j in to_remove \
                                    or paths_radii[i] <= 1.0 or paths_radii[idx_j] <= 1.0:
                                continue  # items are the same or items have already been processed
                            # Brute force approach, shapes are not matched perfectly, so if there are many points close to another
                            # point and distance dosesn't match the points 1-to-1 paths might be merged even though they don't
                            try:  # get bounding boxes for contours from annotation meta-data
                                tile_rect0 = self.metadata[layer['name']]['tile_rect'][i]
                                tile_rect1 = self.metadata[layer['name']]['tile_rect'][idx_j]
                            except KeyError as keyerr:
                                raise KeyError(f"{keyerr.args[0]} - Unspecified metadata for layer {layer['name']}")
                            except IndexError:
                                raise IndexError(
                                    f"Item {i if i > idx_j else idx_j} is missing from layer {layer['name']} metadata")
                            rects_positions, origin_rect, rect_areas = self.check_relative_rect_positions(tile_rect0,
                                                                                                          tile_rect1)  # for tiles
                            if not rects_positions:
                                continue  # do nothing if items are not from touching/overlapping tiles
                            points_near, points_far = (tuple(self.item_points(item0 if origin_rect == 0 else item1)),
                                                       tuple(self.item_points(item0 if origin_rect == 1 else item1)))
                            contour_near = np.array(points_near).astype(np.int32)[:, np.newaxis, :]
                            contour_far = np.array(points_near).astype(np.int32)[:, np.newaxis, :]
                            # points_near stores the points of the top and/or leftmost contour
                            if rects_positions == 'overlap':
                                # if contours overlap, remove overlapping points from the bottom / rightmost contour
                                points_far = self.remove_overlapping_points(points_near, points_far)
                                rects_positions, origin_rect, rect_areas = self.check_relative_rect_positions(
                                    cv2.boundingRect(contour_near),
                                    cv2.boundingRect(contour_far),
                                    eps=10
                                )
                            # not elif as rect_positions could become 'contained' after above
                            if rects_positions == 'contained':  # remove smallest box between the two
                                # if cv2.contourArea(contour_near) >= cv2.contourArea(contour_far):
                                #     to_remove.add(i if origin_rect == 1 else idx_j)  # remove far
                                # else:
                                #     to_remove.add(i if origin_rect == 0 else idx_j)  # remove near
                                continue
                            assert points_far, "Some points must remain from this operation, or rects_positions should have been 'contained'"
                            # noinspection PyBroadException
                            try:
                                # find pairs of points, one in each contour, that may lie at boundary
                                extreme_points = self.find_extreme_points(points_near, points_far,
                                                                          rects_positions, closeness_thresh)
                                if extreme_points:
                                    outer_points = self.get_merged(points_near, points_far, extreme_points)
                                    # make bounding box for new contour
                                    x_out, y_out = (
                                    min(tile_rect0[0], tile_rect1[0]), min(tile_rect0[1], tile_rect1[1]))
                                    w_out = max(tile_rect0[0] + tile_rect0[2],
                                                tile_rect1[0] + tile_rect1[2]) - x_out  # max(x+w) - min(x)
                                    h_out = max(tile_rect0[1] + tile_rect0[3],
                                                tile_rect1[1] + tile_rect1[3]) - y_out  # max(x+w) - min(x)
                                    self.add_item(layer['name'], item0['type'], class_=item0['class'],
                                                  points=outer_points,
                                                  tile_rect=(x_out, y_out, w_out, h_out))
                                    tqdm.tqdm.write(f"Item {i} and {idx_j} were merged ...")
                                    to_remove.add(i)
                                    to_remove.add(idx_j)
                            except Exception:
                                logger.error(f"""
                                 [Layer '{layer['name']}'] iter: #{num_iters} items: {(i, idx_j)} total item num: {len(
                                    items)} merged items:{len(store_items) - len(items)}
                                 [Bounding box] 0 x: {tile_rect0[0]} y: {tile_rect0[1]} w: {tile_rect0[2]} h: {
                                tile_rect0[2]}
                                 [Bounding box] 1 x: {tile_rect1[0]} y: {tile_rect1[1]} w: {tile_rect1[2]} h: {
                                tile_rect1[2]}
                                 Result of rectangle position check: '{rects_positions}', origin: {origin_rect}, areas: {rect_areas}
                                 """)
                                logger.error('Failed.', exc_info=True)
                    for item_idx in sorted(to_remove,
                                           reverse=True):  # remove items that were merged - must start from last index or will return error
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

    def parallel_merge_overlapping_segments(self, closeness_thresh=5.0, max_iter=1, num_workers=4,
                                                log_dir='', timeout=60, centroid_thresh=500.0):
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
                          find_extreme_points=self.find_extreme_points)  # to use functions in subprocess
            logger.info("Begin segment merging ...")
            layer_time = 0.0
            for r, layer in enumerate(self._obj['layers']):
                paths_centroids, paths_radii = self.get_paths_centroids_and_radii(layer['name'])
                if len(paths_centroids) < 2:
                    continue
                centroids = np.array(paths_centroids)
                dist_matrix = cdist(centroids, centroids, 'euclidean')
                changed = True
                num_iters = 0
                items = layer['items']
                logger.info(f"[Layer '{layer['name']}'] Initial number of items is {len(items)}.")
                merged_num = 0 # keep count of change in elements number
                layer_start_time, iter_time, start_time = time.time(), 0.0, time.time()
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
                    input_queue = mp.JoinableQueue(3000)
                    output_queue = mp.Queue(50)  # no need to join
                    # start processes
                    processes = tuple(ItemMerger(i, closeness_thresh, input_queue, output_queue,
                                            to_remove, remove_head, funcs).start() for i in range(num_workers))
                    pair_batch_process_start_time = time.time()  # measure how long it takes to process 500 contours
                    mean_pair_batch_process_time = 0.0
                    to_add = []  # stores items from output queue
                    num_put_to_process, avg_num_pairs_per_contour = 0, 0
                    for i, item0 in enumerate(store_items):  # len(store_items) choose 2
                        # get near contours
                        close_contour_idx = np.where(dist_matrix[i, :] < centroid_thresh)[0]
                        close_contours = tuple(store_items[idx] for idx in close_contour_idx < centroid_thresh)
                        avg_num_pairs_per_contour = avg_num_pairs_per_contour + (
                                len(close_contours) - avg_num_pairs_per_contour) / (i + 1)
                        combinations_num = avg_num_pairs_per_contour * len(store_items)
                        for j, item1 in enumerate(close_contours):
                            idx_j = close_contour_idx[j]  # position of item1 in original annotation
                            if i == idx_j or i in to_remove or idx_j in to_remove:
                                continue  # items are the same or items have already been processed
                            try:
                                tile_rect0 = self.metadata[layer['name']]['tile_rect'][i]
                                tile_rect1 = self.metadata[layer['name']]['tile_rect'][idx_j]
                            except KeyError as keyerr:
                                raise KeyError(f"{keyerr.args[0]} - Unspecified metadata for layer {layer['name']}")
                            except IndexError:
                                raise IndexError(f"Item {i if i > idx_j else idx_j} is missing from layer {layer['name']} metadata")
                            input_queue.put(((i, item0), (idx_j, item1), (tile_rect0, tile_rect1)), timeout=timeout)
                            if num_put_to_process % 500 == 0:
                                mean_pair_batch_process_time = mean_pair_batch_process_time + \
                                (time.time() - pair_batch_process_start_time - mean_pair_batch_process_time) \
                                                               / (num_put_to_process / 500 + 1)  # running average
                                time_left = (combinations_num - num_put_to_process) / 500 * mean_pair_batch_process_time / 60  # in minute
                                elapsed_time = (time.time() - start_time) / 60
                                logger.info(f"Items pairs: processed:{num_put_to_process}/{round(combinations_num)}; {sum(rm > -1 for rm in to_remove)} to remove; " + \
                                            f"duration: {elapsed_time:.2f} mins; eta {time_left:.2f} mins; cpu use: {psutil.cpu_percent()}; available mem: {psutil.virtual_memory().available/1e6:.2f}MB")
                                pair_batch_process_start_time = time.time()  # reset every 500
                                while not output_queue.empty():  # will probably be empty when this point is reached !
                                    data = output_queue.get(timeout=timeout)
                                    if data is None:
                                        sentinel_count += 1
                                        if sentinel_count == num_workers:
                                            break
                                        else:
                                            continue
                                    else:
                                        to_add.append(data)
                                if psutil.virtual_memory().available < 100000:
                                    sys.exit("memory limit exceeded")
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
                        else:
                            to_add.append(data)
                    for data in to_add:
                        (i, idx_j), outer_points, (x_out, y_out, w_out, h_out) = data
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
                        if process is not None:
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
                             'find_extreme_points'))


class ItemMerger(mp.Process):

    def __init__(self, id_, closeness_thresh, input_queue, output_queue,
                 to_remove, remove_head, funcs):
        """
        Handles the merging of items, and returns the merged item to an output queue.
        :param id_:
        :param closeness_thresh:
        :param input_queue:
        :param output_queue:
        :param to_remove:
        :param remove_head:
        :param funcs:
        """
        super().__init__(name='ItemMerger', daemon=True)
        self.id_ = id_
        self.closeness_thresh = closeness_thresh
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
                extreme_points = self.f.find_extreme_points(points_near, points_far,
                                                                        rects_positions, self.closeness_thresh)
                if extreme_points:
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
                    outer_points = self.f.get_merged(points_near, points_far, extreme_points)
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
    parser.add_argument('--centroid_thresh', type=float, default=1000.0)
    parser.add_argument('--closeness_thresh', type=float, default=5.0, help="")
    parser.add_argument('--max_iter', type=int, default=1, help="")
    parser.add_argument('--timeout', type=int, default=60, help="")
    parser.add_argument('--workers', type=int, default=4, help="")
    parser.add_argument('--log_dir', type=str, default='', help="")
    parser.add_argument('--shrink_factor', type=float, default=0.0)
    parser.add_argument('--output_name', type=str, help="")
    parser.add_argument('--sequential', action='store_true')
    args, unparsed = parser.parse_known_args()
    builder = AnnotationBuilder.from_annotation_path(args.annotation_path)
    builder.merge_overlapping_segments(closeness_thresh=args.closeness_thresh, max_iter=args.max_iter,
                                       parallel=not args.sequential, num_workers=args.workers, log_dir=args.log_dir,
                                       timeout=args.timeout, centroid_thresh=args.centroid_thresh)
    if args.shrink_factor:
        builder.shrink_paths(args.shrink_factor)
        print(f"Items were shrunk by {args.shrink_factor}")
    builder.dump_to_json(args.annotation_path.parent, name=args.output_name, rewrite_name=bool(args.output_name))


if __name__ == '__main__':
    main()