"""
Class for translating binary masks into vertice mappings.
Supports:
AIDA format
"""

from itertools import cycle
import copy
import numpy as np
from scipy.stats import mode
from scipy.ndimage import morphology
import skimage.morphology
import cv2
from base.utils import utils  # TODO move tensor2img to object in order to remove this dependency


class MaskConverter:
    """
    Class used to convert ground truth annotation to paths / contours in different
    """

    def __init__(self, dist_threshold=0.1, value_hier=(0, (160, 200), 250), min_contour_area=3000,
                 label_value_map=None, label_interval_map=None, fix_ambiguity=True):
        """
        :param dist_threshold:
        :param value_hier:
        """
        self.dist_threshold = dist_threshold
        self.value_hier = value_hier
        self.min_contour_area = min_contour_area  # use to determine whether to use child contour and to keep contour in the end
        self.fix_ambiguity = fix_ambiguity
        self.by_overlap = False  # option to use bounding boxes to deal with overlapping contours - for deprecated mask_to_contours_all_classes
        self.label_value_map = label_value_map or {
                'epithelium': 200,
                'lumen': 250,
                'background': 0
            }
        self.label_interval_map = label_interval_map or {
            'epithelium': (31, 225),
            'lumen': (225, 250),
            'background': (0, 30)
        }
        assert set(self.label_value_map.keys()) == set(self.label_interval_map.keys()), 'inconsistent annotation classes'
        assert all(isinstance(t, tuple) and len(t) == 2 and t[0] <= t[1] for t in self.label_interval_map.values())
        self.num_classes = len(self.label_value_map)

    def mask_to_contour(self, mask, x_offset=0, y_offset=0, rescale_factor=None):
        """
        Extracts the contours one class at a time
        :param mask:
        :param x_offset:
        :param y_offset:
        :param rescale_factor: rescale image before extracting contours - in case image was shrunk before being fed to segmenation network
        :return:
        """
        mask = utils.tensor2im(mask, segmap=True, num_classes=self.num_classes, visual=False)  # transforms tensors into mask label image
        if rescale_factor:
            mask = cv2.resize(mask.astype(np.uint8), dsize=None, fx=rescale_factor, fy=rescale_factor, interpolation=cv2.INTER_NEAREST)
        if mask.ndim == 3:
            mask = mask[..., 0]
        contours, labels = [], []
        for value in self.label_value_map.values():
            if value == self.label_value_map['background']:
                continue  # don't extract contours for background
            value_binary_mask = self.threshold_by_value(value, mask)
            if self.fix_ambiguity:
                value_binary_mask = self.remove_ambiguity(value_binary_mask, self.dist_threshold)  # makes small glands very small
            if int(cv2.__version__.split('.')[0]) == 3:
                _, value_contours, h = cv2.findContours(value_binary_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_KCOS)
            else:
                value_contours, h = cv2.findContours(value_binary_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_KCOS)
            value_labels = [value] * len(value_contours)
            contours.extend(value_contours)
            labels.extend(value_labels)
            # use RETR_TREE to get full hierarchy (needed for labels)
        assert len(contours) == len(labels)
        good_contours, good_labels, bounding_boxes = [], [], []
        for i, contour in enumerate(contours):
            # filter contours
            is_good = labels[i] is not None and contour.shape[0] > 2 and \
                      cv2.contourArea(contour) > self.min_contour_area
            if is_good:
                x_offset, y_offset = int(x_offset), int(y_offset)
                bounding_box = list(cv2.boundingRect(contour))
                bounding_box[0] += x_offset
                bounding_box[1] += y_offset
                bounding_boxes.append(tuple(bounding_box))
                good_contours.append(contour + np.array((x_offset, y_offset)))  # add offset
                good_labels.append(self.value2label(labels[i]))
        return good_contours, good_labels, bounding_boxes

    def mask_to_contour_all_classes(self, mask, x_offset=0, y_offset=0,
                                    contour_approx_method=cv2.CHAIN_APPROX_TC89_KCOS):
        """
        Extracts the contours once only
        :param mask:
        :param x_offset: x_offset offset
        :param y_offset: y_offset offset
        :param contour_approx_method: standard is method that yields the lowest number of points
        :return: contours of objects in image
        """
        mask = utils.tensor2im(mask, segmap=True, num_classes=self.num_classes, visual=False)  # transforms tensors into mask label image
        if mask.ndim == 3:
            mask = mask[..., 0]
        if self.fix_ambiguity:
            mask = self.remove_ambiguity(mask, self.dist_threshold)  # makes small glands very small
        binary_mask = self.binarize(mask)
        # find contours
        if int(cv2.__version__.split('.')[0]) == 3:
            _, contours, hierarchy = cv2.findContours(binary_mask, cv2.RETR_TREE, contour_approx_method)  # use RETR_TREE to get full hierarchy (needed for labels)
        else:
            contours, hierarchy = cv2.findContours(binary_mask, cv2.RETR_TREE,
                                                   contour_approx_method)  # use RETR_TREE to get full hierarchy (needed for labels)
        labels = self.get_contour_labels_by_overlap(mask, contours) if self.by_overlap \
            else self.get_contours_labels(mask, contours, hierarchy)
        assert len(contours) == len(labels)
        # remove short contours and contours that are too small,
        # then add offset - NB: coordinates are in form (y,x)
        good_contours, good_labels, bounding_boxes = [], [], []
        for i, contour in enumerate(contours):
            # filter contours
            is_good = labels[i] is not None and contour.shape[0] > 2 and \
                             cv2.contourArea(contour) > self.min_contour_area
            if is_good:
                bounding_boxes.append(cv2.boundingRect(contour))
                good_contours.append(contour + np.array((x_offset, y_offset)))  # add offset
                good_labels.append(self.value2label(labels[i]))
        return good_contours, good_labels, bounding_boxes

    def threshold_by_value(self, value, mask):
        """
        Use label hierarchy to threshold values
        :param value:
        :return:
        """
        mask = mask.copy()
        for value_level in reversed(self.value_hier):
            try:
                for v in value_level:
                    mask[mask == v] = 1
                if value == v:
                    break
            except TypeError:
                mask[mask == value_level] = 1
                if value == value_level:
                    break
        mask[mask != 1] = 0
        return mask

    def binarize(self, mask):
        """
        Uses self.value_hier to determine which label should be inside which other label.
        The last level is the innermost one
        If different label values are on the same level, this should be passed as a tuple.
        First label should be background and is assigned a value of 0
        :param mask:
        :return:
        """
        mask = mask.copy()
        if mask.ndim == 3:
            mask = mask[..., 0]
        # turn different labels (rgb value * [1,1,1]) into alternating black and white regions for hierarchy extraction
        for label, alter in zip(self.value_hier, cycle([0, 1])):
            if isinstance(label, tuple):  # account for different labels on the same level
                for l_label in label:
                    mask[mask == l_label] = alter
            else:
                mask[mask == label] = alter
        return mask

    @staticmethod
    def get_inner_values(mask, contour):
        """
        :param mask:
        :param contour:
        :return: mask values within contour
        """
        assert type(contour) is np.ndarray, "Must be an Nx1x2    numpy array "
        assert contour.shape[0] >= 3, f"Don't do small contours (shape[0] = {contour.shape[0]})"
        assert contour.ndim > 2, f"Don't do ill-defined contours (ndim = {contour.ndim})"
        x, y, w, h = cv2.boundingRect(contour)
        # use bounding box so that label mask must be created only for size of contour and not for whole tile / area
        contour_shift = contour.squeeze() - np.array((x, y))  # shift so that contour is relative to (0,0)
        contour_shift = np.expand_dims(contour_shift, axis=1)
        masklet = mask[y:y + h, x:x + w]
        # Create a mask image that contains the contour filled in
        cimg = np.zeros((w, h))
        # print(contour_shift[0])
        cv2.drawContours(cimg, [contour_shift], 0, color=255,
                         thickness=-1)  # fills the whole area inside contour with colour
        # Access the image pixels and create a 1D numpy array then add to list
        pts = np.where(cimg == 255)
        values_within_contour = set(masklet[pts[0], pts[1]])
        mode_value = mode(masklet[pts[0], pts[1]])[0][0]
        return values_within_contour, mode_value

    def get_contours_labels(self, mask, contours, hierarchy):
        """
        Baseline - uses cv2 hierarchy
        :param mask:
        :param contours:
        :param hierarchy:
        :return: label for each contour | =None if contour is too small / it's only 2 points
        """
        contours_labels = []
        for i, contour in enumerate(contours):
            if contour.shape[0] < 3 or cv2.contourArea(contour) < self.min_contour_area:
                contours_labels.append(None)
                continue
            values_within_contour, parent_mode = self.get_inner_values(mask, contour)
            next_, previous, first_child, parent = hierarchy[0][i]
            child = first_child
            if len(values_within_contour) != 1:
                inner_contour_values = set()
                while child != -1:
                    # removes label values of 1 level deep children
                    if contours[child].size > 0 and contours[child].shape[0] > 1 \
                            and cv2.contourArea(contour[child]) > self.min_contour_area:
                        child_values, child_mode = self.get_inner_values(mask, contours[child])
                        inner_contour_values |= child_values
                    child, _, __, ___ = hierarchy[0][child]
                if len(inner_contour_values) >= len(values_within_contour) \
                        and inner_contour_values != values_within_contour:
                    # get the lowest in the hierarchy (e.g. lumen)
                    for value in reversed(self.value_hier):
                        if value in inner_contour_values:
                            inner_contour_values = {value}
                            break
                values_within_contour -= inner_contour_values
            if len(values_within_contour) != 1:
                if first_child != -1:
                    try:
                        if parent_mode not in self.value_hier[-1]:  # if not in last value (deals with multiple values on same level)
                            values_within_contour = {parent_mode}
                        else:
                            try:
                                values_within_contour = {self.value_hier[-2][-1]}
                            except TypeError:
                                values_within_contour = {self.value_hier[-2]}
                    except TypeError:
                        if parent_mode != self.value_hier[-1]:
                            values_within_contour = {parent_mode}
                        else:
                            try:
                                values_within_contour = {self.value_hier[-2][-1]}
                            except TypeError:
                                values_within_contour = {self.value_hier[-2]}
                else:
                    # in case contour has no children
                    try:
                        values_within_contour = {self.value_hier[-1][-1]}
                    except TypeError:
                        values_within_contour = {self.value_hier[-1]}

            assert len(values_within_contour) == 1, "only one value per contour must remain"
            contours_labels.append(int(next(iter(values_within_contour))))
        return contours_labels

    def get_contour_labels_by_overlap(self, mask, contours):
        """
        Check what bounding contours are contained within each other
        :param mask:
        :param contours:
        :return:
        """
        contour_bbs = list(cv2.boundingRect(contour) for i, contour in enumerate(contours))
        overlap_struct = []
        for parent_bb in contour_bbs:
            # Find out each contour's children (overlapping)
            overlap_vector = []
            for child_bb in contour_bbs:
                contained = self.check_bounding_boxes_overlap(parent_bb, child_bb)
                if parent_bb == child_bb:  # don't do anything for same box
                    overlap_vector.append(False)
                elif contained:
                    overlap_vector.append(True)
                else:
                    overlap_vector.append(False)
            overlap_struct.append(overlap_vector)
        contours_labels = []
        for contour, overlap_vector in zip(contours, overlap_struct):
            if contour.shape[0] < 3 or cv2.contourArea(contour) < self.min_contour_area:
                contours_labels.append(None)
                continue
            values_within_contour, parent_mode = self.get_inner_values(mask, contour)
            if len(values_within_contour) > 1:
                # pick value that is different from inner values of inner contours
                inner_contour_values = set()
                for i, contained in enumerate(overlap_vector):
                    if contained and contours[i].shape[0] >= 3 and cv2.contourArea(contour) >= self.min_contour_area :
                        contained_values, child_mode = self.get_inner_values(mask, contours[i])
                        inner_contour_values |= contained_values  # union and update
                if len(inner_contour_values) >= len(values_within_contour) \
                        and inner_contour_values != values_within_contour:
                    # get the lowest in the hierarchy (e.g. lumen)
                    for value in reversed(self.value_hier):
                        if value in inner_contour_values:
                            inner_contour_values = {value}
                            break
                values_within_contour -= inner_contour_values
            if len(values_within_contour) != 1:
                # in case above doesn't yield only one value (could leave 0 or >1)
                if any(overlap_vector):
                    # check whether contour has inner contours - if so, assign higher value
                    try:
                        if parent_mode not in self.value_hier[-1]:  # if not in last value (deals with multiple values on same level)
                            values_within_contour = {parent_mode}
                        else:
                            try:
                                values_within_contour = {self.value_hier[-2][-1]}
                            except TypeError:
                                values_within_contour = {self.value_hier[-2]}
                    except TypeError:
                        if parent_mode != self.value_hier[-1]:
                            values_within_contour = {parent_mode}
                        else:
                            try:
                                values_within_contour = {self.value_hier[-2][-1]}
                            except TypeError:
                                values_within_contour = {self.value_hier[-2]}
                else:
                    # in case contour mode is lowest level in hier or contour has no children
                    try:
                        values_within_contour = {self.value_hier[-1][-1]}
                    except TypeError:
                        values_within_contour = {self.value_hier[-1]}
            assert len(values_within_contour) == 1, "Only one value per contour must remain"
            contours_labels.append(int(next(iter(values_within_contour))))
        return contours_labels

    @staticmethod
    def check_bounding_boxes_overlap(parent_bb, child_bb):
        x0, y0, w0, h0 = parent_bb
        x1, y1, w1, h1 = child_bb
        x_w0, y_h0, x_w1, y_h1 = x0 + w0, y0 + h0, x1 + w1, y1 + h1
        return (x0 <= x1 <= x_w0 and
                x0 <= x_w1 <= x_w0 and
                y0 <= y1 <= y_h0 and
                y0 <= y_h1 <= y_h0)

    @staticmethod
    def remove_ambiguity(mask, dist_threshold=2.0, small_area_factor=0.4):
        """
        Takes HxWx3 image with identical channels, or HxW image
        :param mask:
        :param dist_threshold: multiplied by mode of peaks in distance transform -- e,g, 2.0 is twofold the average peak
        :param small_area_factor:
        :return:
        """
        mask = copy.deepcopy(mask)
        if mask.ndim == 3:
            mask_1c = mask[..., 0]  # need to keep original mask as watershed wants 3 channels image
        else:
            mask_1c = mask
            mask = np.tile(mask[..., np.newaxis], (1, 1, 3))
        # noise removal
        kernel = np.ones((3, 3), np.uint8)
        opening = cv2.morphologyEx(mask_1c, cv2.MORPH_OPEN, kernel, iterations=2)
        # refine -ve area (includes background)
        refined_bg = cv2.dilate(opening, kernel, iterations=3)
        # refine foreground area
        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        # get mode of maxima in image, to use as reference for threshold (more invariant than absolute max in tile)
        maxima = skimage.morphology.local_maxima(dist_transform, indices=True)
        maxima = np.stack(maxima, axis=0).T
        values_at_maxima = np.array(list(dist_transform[y, x] for y, x in maxima))
        mode_of_maxima = mode(values_at_maxima)[0]
        mode_of_maxima = mode_of_maxima.item(0) if mode_of_maxima.size > 0 else 255
        # threshold using distance transform
        ret, refined_fg = cv2.threshold(dist_transform,
                                        min(dist_threshold * mode_of_maxima, 0.1 * dist_transform.max()), 255, 0)
        refined_fg = np.uint8(refined_fg)
        # finding unknown region
        unknown = cv2.subtract(refined_bg, refined_fg)
        # marker labelling
        ret, markers = cv2.connectedComponents(refined_fg)
        markers = markers + 1
        # mark the region of unknown with zero
        markers[unknown == 250] = 0
        # watershed
        markers = cv2.watershed(mask, markers)
        # threshold out boundaries and background (-1 and 0 respectively)
        markers = cv2.morphologyEx(cv2.medianBlur(markers.astype(np.uint8), 3), cv2.MORPH_OPEN, kernel, iterations=2)
        unambiguous = np.uint8(cv2.medianBlur(markers.astype(np.uint8), 3) > 1) * 255
        # filled holes if any in larger objects
        unambiguous = morphology.binary_fill_holes(unambiguous)
        # remove small objects
        unambiguous = skimage.morphology.remove_small_objects(unambiguous, min_size=mask.shape[0] * small_area_factor)
        return unambiguous.astype(np.uint8)

    def value2label(self, value):
        label = None
        for l, (b1, b2) in self.label_interval_map.items():
            if b1 <= value <= b2:
                if not label:
                    label = l
                    bounds = (b1, b2)
                else:
                    raise ValueError(f"Overlapping interval bounds ({b1}, {b2}) and {bounds}, for {l} and {label}")
        if not label:
            raise ValueError(f'Value {value} is not withing any interval')
        return label

    def label2value(self, label):
        return self.label_value_map[label]

    @staticmethod
    def contour_to_mask(contour, value=255, shape=()):
        """
        Convert a contour to the corresponding max - mask is
        :param contour:
        :param value:
        :param shape:
        :return:
        """
        if not shape:
            shape = (contour.max(0) - contour.min(0), contour.max(1) - contour.min(1))  # dimension of contour
        mask = np.ones(shape)
        cv2.drawContours(mask, [contour], -1, value, thickness=-1)  # thickness=-1 fills the entire area inside
        return mask


# class PaperJSPathEmulator:
#     """
#     Can use __dict__ property to generate same dict as would paper.Path objects in JS
#     """
#
#     @classmethod
#     def encode_path_array(cls, path_emulator):
#         if isinstance(path_emulator, cls):
#             return ['Path', path_emulator.__dict__]
#         else:
#             raise TypeError(f"Object of type '{path_emulator.__class__.__name__}' is not JSON serializable")
#
#     def __init__(self, contour, label, handles=False, stroke_color=(0, 0, 1)):
#         contour = contour.squeeze().tolist()  # work with opencv contour output (numpy array)
#         if handles:
#             self.segments = [[coords, [0, 0], [0, 0]] for coords in contour]
#         else:
#             self.segments = [[coords] for coords in contour]
#         # attributes in paper.js Path obj
#         self.applyMatrix = True
#         self.closed = True
#         self.strokeColor = list(stroke_color)
#         self.label = label
