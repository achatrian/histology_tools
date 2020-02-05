import numpy as np
import cv2
from skimage.morphology import remove_small_objects


def is_hne(image, threshold=0.4, sat_thresh=20, small_obj_size_factor=1/5):
    r"""Returns true if slide contains tissue or just background
    Problem: doesn't work if tile is completely white because of normalization step"""
    if image.shape[-1] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(float)
    # saturation check
    sat_mean = hsv_image[..., 1].mean()
    empirical = hsv_image.prod(axis=2)  # found by Ka Ho to work
    empirical = empirical /np.max(empirical) * 255  # found by Ka Ho to work
    kernel = np.ones((20, 20), np.uint8)
    ret, _ = cv2.threshold(empirical.astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = cv2.morphologyEx((empirical > ret).astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    mask = remove_small_objects(mask.astype(bool), min_size=image.shape[0] * small_obj_size_factor)
    return mask.mean() > threshold and sat_mean >= sat_thresh