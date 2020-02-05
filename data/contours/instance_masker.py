from functools import partial
from typing import Union
import warnings
import numpy as np
from data.contours import contour_to_mask, find_overlap


class InstanceMasker:
    r"""Use to return multilabel masks for contours."""

    def __init__(self, slide_contours: Union[dict, tuple], outer_label: str, label_values: dict,
                shape=(), contour_to_mask=contour_to_mask):
        r"""
        :param slide_contours: dict: layer name -> contours or (contours, labels)
        :param outer_label: contours with this label are assumed to be outermost: masks are built from these contours
        :param label_values: dict: layer name -> pixel value
        :param shape:
        :param contour_to_mask:
        """
        self.label_values = label_values
        self.outer_label = outer_label
        self.shape = shape
        self.contour_to_mask = partial(contour_to_mask, shape=shape) if shape else contour_to_mask
        if isinstance(slide_contours, dict):
            self.contours, self.labels = [], []
            for layer_name, layer_contours in slide_contours.items():
                self.contours.extend(layer_contours)
                self.labels.extend([layer_name] * len(layer_contours))
        else:
            self.contours, self.labels = slide_contours
        # remove contours that are empty or have a single point
        self.checked_contours_indices = tuple(i for i, contour in enumerate(self.contours) if contour.size > 2)
        self.contours = [contour for i, contour, in enumerate(self.contours) if i in self.checked_contours_indices]
        self.labels = [label for i, label, in enumerate(self.labels) if i in self.checked_contours_indices]
        if not set(label_values.keys()) >= set(self.labels):
            warnings.warn(f"Pixel value for {tuple(set(self.labels) - set(label_values.keys()))} is unspecified, ignoring these labels.")
        # outer label specifies the contours the interface gives access to
        self.outer_contours_indices = tuple(i for i, label in enumerate(self.labels) if label == outer_label)
        self.overlap_struct, self.contours, self.bounding_boxes, self.labels = find_overlap(slide_contours)

    def __len__(self):
        return len(self.outer_contours_indices)

    def __getitem__(self, index, shape=None) -> np.array:
        if index > len(self):
            raise ValueError("Index exceeds number of outer contours")
        outer_index = self.outer_contours_indices[index]
        overlap_vect = self.overlap_struct[outer_index]
        mask = contour_to_mask(self.contours[outer_index], value=self.label_values[self.labels[outer_index]],
                               shape=shape)
        x_parent, y_parent, w_parent, h_parent = self.bounding_boxes[outer_index]
        components = {'parent_contour': self.contours[outer_index],
                      'children_contours': [],
                      'parent_label': self.labels[outer_index],
                      'children_labels': []}
        for child_index in overlap_vect:
            if not self.labels[child_index] in self.label_values:
                continue  # skip all contours for whom a paint-in value is unspecified
            x_child, y_child, w_child, h_child = self.bounding_boxes[child_index]
            if h_parent * w_parent > h_child * w_child:  # if parent bigger than child write on previous mask
                mask = contour_to_mask(self.contours[child_index], mask=mask, mask_origin=(x_parent, y_parent),
                                       value=self.label_values[self.labels[child_index]])
                components['children_contours'].append(self.contours[child_index])
                components['children_labels'].append(self.labels[child_index])
        return mask, components

    def get_shaped_mask(self, index, shape):
        return self.__getitem__(index, shape=shape)

    @property
    def outer_contours(self):
        return list(self.contours[i] for i in self.outer_contours_indices)
