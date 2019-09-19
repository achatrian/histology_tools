import numpy as np
import torch


# Converts a Tensor into an images array (numpy)
# |imtype|: the desired type of the converted numpy array
def tensor2im(input_image, segmap=False, num_classes=3, imtype=np.uint8, visual=True):
    r"""
    Converts images to tensor for visualisation purposes
    :param input_image:
    :param segmap:
    :param num_classes
    :param imtype:
    :param visual: whether output is destined for visualization or processing (3c vs 1c)
    :return: images
    """
    if isinstance(input_image, torch.Tensor):
        image_tensor = input_image.data
    else:
        return input_image
    image_numpy = image_tensor.cpu().float().numpy()  # taking the first images only NO MORE
    if image_numpy.shape[0] == 1:
        image_numpy = np.tile(image_numpy, (3, 1, 1))

    if segmap:
        image_numpy = segmap2img(image_numpy, num_classes=num_classes)
    else:
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
        # for segmentation maps with four classes
    if image_numpy.ndim == 2 and visual:
        image_numpy = image_numpy[:, :, np.newaxis].repeat(3, axis=2)
    return image_numpy.astype(imtype)


def segmap2img(segmap, num_classes=None):
    r"""
    Converts segmentation maps in a one-class-per-channel or one-value-per-class encodings into visual images
    :param segmap: the segmentation map to convert
    :param num_classes: number of classes in map. If the map is single-channeled, num_classes must be passed and be nonzero
    :return:
    """
    if len(segmap.shape) > 2:
        # multichannel segmap, one channel per class
        if segmap.shape[0] < segmap.shape[1] and segmap.shape[0] < segmap.shape[2]:
            segmap = segmap.transpose(1, 2, 0)
        image = np.argmax(segmap, axis=2)
        if segmap.shape[2] == 4:
            image[image == 1] = 160
            image[image == 2] = 200
            image[image == 3] = 250
        elif segmap.shape[2] == 3:
            image[image == 1] = 200
            image[image == 2] = 250
        elif segmap.shape[2] == 2:
            image[image == 1] = 250
        else:
            raise ValueError("Conversion of map to images not supported for shape {}".format(segmap.shape))
    elif num_classes:
        num_labels = len(np.unique(segmap))
        if num_labels > num_classes:
            raise ValueError(f"More labels than classes in segmap ({num_labels} > {num_classes}")
        if num_classes == 2:
            segmap *= 2
        elif num_classes == 3:
            segmap[segmap == 1] = 200
            segmap[segmap == 2] = 250
        elif num_classes == 4:
            segmap[segmap == 1] = 160
            segmap[segmap == 2] = 200
            segmap[segmap == 3] = 250
        else:
            raise NotImplementedError(f"Can't handle {num_classes} classes")
        image = segmap
    else:
        raise ValueError('For single channel segmap, num_classes must be > 0')
    return image
