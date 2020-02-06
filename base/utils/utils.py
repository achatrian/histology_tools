from __future__ import print_function
from pathlib import Path
import time
import socket
import re
from collections import OrderedDict
from torch import nn
from argparse import ArgumentTypeError
import numpy as np
import cv2
# TODO remove unused functions and clean up existing and used ones !


def str_is_int(s):
    r"""
    Check if string is convertable to an integer
    """
    try:
        int(s)
        return True
    except ValueError:
        return False


def str2bool(v):
    r"""
    Use with argparse to convert string to bool
    """
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ArgumentTypeError('Boolean value expected.')


def get_time_stamp():
    date_string = time.strftime("%Y_%m_%d_%H_%M_%S")
    return date_string


def split_options_string(opt_string, splitter=','):
    opts = opt_string.split(f'{splitter}')
    return [int(opt) if str_is_int(opt) else opt for opt in opts]


def on_cluster():
    hostname = socket.gethostname()
    match1 = re.search("jalapeno(\w\w)?.fmrib.ox.ac.uk", hostname)
    match2 = re.search("cuda(\w\w)?.fmrib.ox.ac.uk", hostname)
    match3 = re.search("login(\w\w)?.cluster", hostname)
    match4 = re.search("gpu(\w\w)?", hostname)
    match5 = re.search("compG(\w\w\w)?", hostname)
    match6 = re.search("rescomp(\w)?", hostname)
    return bool(match1 or match2 or match3 or match4 or match5)


def bytes2human(n):
    # http://code.activestate.com/recipes/578019
    # >>> bytes2human(10000)
    # '9.8K'
    # >>> bytes2human(100001221)
    # '95.4M'
    symbols = ('K', 'M', 'G', 'T', 'P', 'E', 'Z', 'Y')
    prefix = dict()
    for i, s in enumerate(symbols):
        prefix[s] = 1 << (i + 1) * 10
    for s in reversed(symbols):
        if n >= prefix[s]:
            value = float(n) / prefix[s]
            return '{:.2f}{}B'.format(value, s)
    return "{}B".format(n)


#### torch

def initialize_weights(*models):
    for model in models:
        for module in model.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()


class AverageMeter(object):
    __slots__ = ['val', 'avg', 'sum', 'count']

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0.0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def colorize(gt):
    if gt.shape[0] > 1:
        if gt.min() < 0 or gt.max() > 1:  #if logit
            if np.any(gt > 500): gt /= gt.max()
            gt = np.exp(gt) / np.repeat(np.exp(gt).sum(axis=0)[np.newaxis,...], gt.shape[0], axis=0)
        gt = np.round(gt)

        if gt.shape[0] == 3:
            r = np.floor((gt[0, ...] == 1) * 255)
            g = np.floor((gt[1, ...] == 1) * 255)
            b = np.floor((gt[2, ...] == 1) * 255)
        elif gt.shape[0] == 2:
            r = np.floor((gt[1, ...] == 1) * 255)
            g = np.floor((gt[0, ...] == 1) * 255)
            b = np.zeros(gt[0, ...].shape)
        elif gt.shape[0] == 1:
            r,g,b = gt*255, gt*255, gt*255
        else:
            raise NotImplementedError
    else:
        if gt.min() < 0 or gt.max() > 1:
            gt = 1/(1+np.exp(-gt))
        gt = np.round(gt)[0,...]
        r, g, b = gt*255, gt*255, gt*255
    gt_colorimg = np.stack([r, g, b], axis=2).astype(np.uint8)
    return gt_colorimg


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


# TODO fix this if needed - not updated in a while
def img2segmap(gts, return_tensors=False, size=128):
    """
    !!! Currently only works for pix2pix version !!!
    :param gts:
    :param return_tensors:
    :param size:
    :return:
    """

    def denormalize(value):
        # undoes normalization that is applied in pix2pix aligned dataset
        return value * 0.5 + 0.5

    gts = denormalize(gts)

    if isinstance(gts, torch.Tensor):
        gts = gts.cpu().numpy()
    if gts.ndim == 3:
        gts = gts.transpose(1, 2, 0)
        gts = [gts]  # to make function work for single images too
    else:
        gts = gts.transpose(0, 2, 3, 1)

    gt_store, label_store = [], []
    for gt in gts:
        if gt.shape[0:2] != (size,)*2:
            gt = cv2.resize(gt, dsize=(size,)*2)
        label = stats.mode(gt[np.logical_and(gt > 0, gt != 250)], axis=None)[
            0]  # take most common class over gland excluding lumen
        if label.size > 0:
            label = int(label)
            if np.isclose(label*255, 160):
                label = 0
            elif np.isclose(label*255, 200):
                label = 1
        else:
            label = 0.5

        gt[np.isclose(gt*255, 160, atol=45)] = 40/255  # to help get better map with irregularities introduced by augmentation

        # Normalize as wh en training network:

        gt = gt[:, :, 0]
        gt = np.stack((np.uint8(np.logical_and(gt >= 0, gt < 35/255)),
                       np.uint8(np.logical_and(gt >= 35/255, gt < 45/255)),
                       np.uint8(np.logical_and(gt >= 194/255, gt < 210/255)),
                       np.uint8(np.logical_and(gt >= 210/255, gt <= 255/255))), axis=2)

        if return_tensors:
            gt = torch.from_numpy(gt.transpose(2, 0, 1)).float()
            label = torch.tensor(label).long()
        gt_store.append(gt)
        label_store.append(label)

    gts = torch.stack(gt_store, dim=0) if return_tensors else np.stack(gt_store, axis=0)
    labels = torch.stack(label_store, dim=0) if return_tensors else np.stack(label_store, axis=0)
    return gts, labels


def diagnose_network(net, name='network'):
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)


def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)


def summary(model, input_size, device, batch_size=-1):
    """
    Prints out a detailed summary of the pytorch model.
    From: https://github.com/sksq96/pytorch-summary
    :param model:
    :param input_size:
    :param batch_size:
    :param device:
    :return:
    """

    def register_hook(module):

        def hook(module, input, output):
            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            module_idx = len(summary)

            m_key = "%s-%i" % (class_name, module_idx + 1)
            summary[m_key] = OrderedDict()
            summary[m_key]["input_shape"] = list(input[0].size())
            summary[m_key]["input_shape"][0] = batch_size
            if isinstance(output, (list, tuple)):
                summary[m_key]["output_shape"] = [
                    [-1] + list(o.size())[1:] for o in output
                ]
            else:
                summary[m_key]["output_shape"] = list(output.size())
                summary[m_key]["output_shape"][0] = batch_size

            params = 0
            if hasattr(module, "weight") and hasattr(module.weight, "size"):
                params += torch.prod(torch.LongTensor(list(module.weight.size())))
                summary[m_key]["trainable"] = module.weight.requires_grad
            if hasattr(module, "bias") and hasattr(module.bias, "size"):
                params += torch.prod(torch.LongTensor(list(module.bias.size())))
            summary[m_key]["nb_params"] = params

        if (
                not isinstance(module, nn.Sequential)
                and not isinstance(module, nn.ModuleList)
                and not (module == model)
        ):
            hooks.append(module.register_forward_hook(hook))

    device = device.lower()
    assert device in [
        "cuda",
        "cpu",
    ], "Input device is not valid, please specify 'cuda' or 'cpu'"

    if device == "cuda" and torch.cuda.is_available():
        dtype = torch.cuda.FloatTensor
        model.cuda()
    else:
        dtype = torch.FloatTensor

    # multiple inputs to the network
    if isinstance(input_size, tuple):
        input_size = [input_size]

    # batch_size of 2 for batchnorm
    x = [torch.rand(2, *in_size).type(dtype) for in_size in input_size]
    # print(type(x[0]))

    # create properties
    summary = OrderedDict()
    hooks = []

    # register hook
    model.apply(register_hook)

    # make a forward pass
    model(*x)

    # remove these hooks
    for h in hooks:
        h.remove()

    print("----------------------------------------------------------------")
    line_new = "{:>20}  {:>25} {:>15}".format("Layer (type)", "Output Shape", "Param #")
    print(line_new)
    print("================================================================")
    total_params = 0
    total_output = 0
    trainable_params = 0
    for layer in summary:
        # input_shape, output_shape, trainable, nb_params
        line_new = "{:>20}  {:>25} {:>15}".format(
            layer,
            str(summary[layer]["output_shape"]),
            "{0:,}".format(summary[layer]["nb_params"]),
        )
        total_params += summary[layer]["nb_params"]
        total_output += np.prod(summary[layer]["output_shape"])
        if "trainable" in summary[layer]:
            if summary[layer]["trainable"] == True:
                trainable_params += summary[layer]["nb_params"]
        print(line_new)

    # assume 4 bytes/number (float on cuda).
    total_input_size = abs(np.prod(input_size) * batch_size * 4. / (1024 ** 2.))
    total_output_size = abs(2. * total_output * 4. / (1024 ** 2.))  # x2 for gradients
    total_params_size = abs(total_params.numpy() * 4. / (1024 ** 2.))
    total_size = total_params_size + total_output_size + total_input_size

    print("================================================================")
    print("Total params: {0:,}".format(total_params))
    print("Trainable params: {0:,}".format(trainable_params))
    print("Non-trainable params: {0:,}".format(total_params - trainable_params))
    print("----------------------------------------------------------------")
    print("Input size (MB): %0.2f" % total_input_size)
    print("Forward/backward pass size (MB): %0.2f" % total_output_size)
    print("Params size (MB): %0.2f" % total_params_size)
    print("Estimated Total Size (MB): %0.2f" % total_size)
    print("----------------------------------------------------------------")
    # return summary


# Saliency maps utils
"""
Created on Thu Oct 21 11:09:09 2017

@author: Utku Ozbulak - github.com/utkuozbulak
"""
import os
import copy
import numpy as np
from PIL import Image
import matplotlib.cm as mpl_color_map

import torch
from torch.autograd import Variable
from torchvision import models


def convert_to_grayscale(im_as_arr):
    """
        Converts 3d images to grayscale

    Args:
        im_as_arr (numpy arr): RGB images with shape (D,W,H)

    returns:
        grayscale_im (numpy_arr): Grayscale images with shape (1,W,D)
    """
    grayscale_im = np.sum(np.abs(im_as_arr), axis=0)
    im_max = np.percentile(grayscale_im, 99)
    im_min = np.min(grayscale_im)
    grayscale_im = (np.clip((grayscale_im - im_min) / (im_max - im_min), 0, 1))
    grayscale_im = np.expand_dims(grayscale_im, axis=0)
    return grayscale_im


def save_gradient_images(gradient, file_path):
    """
        Exports the original gradient images

    Args:
        gradient (np arr): Numpy array of the gradient with shape (3, 224, 224)
        file_name (str): File name to be exported
        save_dir (str): path to saving location
    """
    # Normalize
    gradient = gradient - gradient.min()
    gradient /= gradient.max()
    # Save images
    file_path = Path(file_path).with_suffix('.png')
    file_path.parent.mkdir(exist_ok=True)
    save_image(gradient, file_path)


def save_class_activation_images(org_img, activation_map, file_name):
    """
        Saves cam activation map and activation map on the original images

    Args:
        org_img (PIL example_grid): Original images
        activation_map (numpy arr): Activation map (grayscale) 0-255
        file_name (str): File name of the exported images
    """
    if not os.path.exists('../results'):
        os.makedirs('../results')
    # Grayscale activation map
    heatmap, heatmap_on_image = apply_colormap_on_image(org_img, activation_map, 'hsv')
    # Save colored heatmap
    path_to_file = os.path.join('../results', file_name+'_Cam_Heatmap.png')
    print(np.max(heatmap))
    save_image(heatmap, path_to_file)
    # Save heatmap on iamge
    print()
    print(np.max(heatmap_on_image))
    path_to_file = os.path.join('../results', file_name+'_Cam_On_Image.png')
    save_image(heatmap_on_image, path_to_file)
    # Save grayscale heatmap
    print()
    print(np.max(activation_map))
    path_to_file = os.path.join('../results', file_name+'_Cam_Grayscale.png')
    save_image(activation_map, path_to_file)


def apply_colormap_on_image(org_im, activation, colormap_name):
    """
        Apply heatmap on images
    Args:
        org_img (PIL example_grid): Original images
        activation_map (numpy arr): Activation map (grayscale) 0-255
        colormap_name (str): Name of the colormap
    """
    # Get colormap
    color_map = mpl_color_map.get_cmap(colormap_name)
    no_trans_heatmap = color_map(activation)
    # Change alpha channel in colormap to make sure original images is displayed
    heatmap = copy.copy(no_trans_heatmap)
    heatmap[:, :, 3] = 0.4
    heatmap = Image.fromarray((heatmap*255).astype(np.uint8))
    no_trans_heatmap = Image.fromarray((no_trans_heatmap*255).astype(np.uint8))

    # Apply heatmap on iamge
    heatmap_on_image = Image.new("RGBA", org_im.size)
    heatmap_on_image = Image.alpha_composite(heatmap_on_image, org_im.convert('RGBA'))
    heatmap_on_image = Image.alpha_composite(heatmap_on_image, heatmap)
    return no_trans_heatmap, heatmap_on_image


def save_image(im, path):
    """
        Saves a numpy matrix of shape D(1 or 3) x W x H as an images
    Args:
        im_as_arr (Numpy array): Matrix of shape DxWxH
        path (str): Path to the images

    TODO: Streamline images saving, it is ugly.
    """
    if isinstance(im, np.ndarray):
        if len(im.shape) == 2:
            im = np.expand_dims(im, axis=0)
            print('A')
            print(im.shape)
        if im.shape[0] == 1:
            # Converting an images with depth = 1 to depth = 3, repeating the same values
            # For some reason PIL complains when I want to save channel images as jpg without
            # additional format in the .save()
            print('B')
            im = np.repeat(im, 3, axis=0)
            print(im.shape)
            # Convert to values to range 1-255 and W,H, D
        # A bandaid fix to an issue with gradcam
        if im.shape[0] == 3 and np.max(im) == 1:
            im = im.transpose(1, 2, 0) * 255
        elif im.shape[0] == 3 and np.max(im) > 1:
            im = im.transpose(1, 2, 0)
        elif im.shape[2] == 3:
            if np.max(im) == 1:
                im = im * 255
        else:
            raise ValueError(f"Invalid array dimensions {im.shape} for images data")
        im = Image.fromarray(im.astype(np.uint8))
    im.save(path)


def preprocess_image(pil_im, resize_im=True):
    """
        Processes images for CNNs

    Args:
        PIL_img (PIL_img): Image to process
        resize_im (bool): Resize to 224 or not
    returns:
        im_as_var (torch variable): Variable that contains processed float tensor
    """
    # mean and std list for channels (Imagenet)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    # Resize images
    if resize_im:
        pil_im.thumbnail((512, 512))
    im_as_arr = np.float32(pil_im)
    im_as_arr = im_as_arr.transpose(2, 0, 1)  # Convert array to D,W,H
    # Normalize the channels
    for channel, _ in enumerate(im_as_arr):
        im_as_arr[channel] /= 255
        im_as_arr[channel] -= mean[channel]
        im_as_arr[channel] /= std[channel]
    # Convert to float tensor
    im_as_ten = torch.from_numpy(im_as_arr).float()
    # Add one more channel to the beginning. Tensor shape = 1,3,224,224
    im_as_ten.unsqueeze_(0)
    # Convert to Pytorch variable
    im_as_var = Variable(im_as_ten, requires_grad=True)
    return im_as_var


def recreate_image(im_as_var):
    """
        Recreates images from a torch variable, sort of reverse preprocessing
    Args:
        im_as_var (torch variable): Image to recreate
    returns:
        recreated_im (numpy arr): Recreated images in array
    """
    reverse_mean = [-0.485, -0.456, -0.406]
    reverse_std = [1/0.229, 1/0.224, 1/0.225]
    recreated_im = copy.copy(im_as_var.data.numpy()[0])
    for c in range(3):
        recreated_im[c] /= reverse_std[c]
        recreated_im[c] -= reverse_mean[c]
    recreated_im[recreated_im > 1] = 1
    recreated_im[recreated_im < 0] = 0
    recreated_im = np.round(recreated_im * 255)

    recreated_im = np.uint8(recreated_im).transpose(1, 2, 0)
    return recreated_im


def get_positive_negative_saliency(gradient):
    """
        Generates positive and negative saliency maps based on the gradient
    Args:
        gradient (numpy arr): Gradient of the operation to visualize

    returns:
        pos_saliency ( )
    """
    pos_saliency = (np.maximum(0, gradient) / gradient.max())
    neg_saliency = (np.maximum(0, -gradient) / -gradient.min())
    return pos_saliency, neg_saliency


def overlay_grids(example_grid, gradient_grid, threshold=0.2):
    r"""overlay images and gradient grids by alpha blending"""
    # https://docs.opencv.org/trunk/d0/d86/tutorial_py_image_arithmetics.html
    # overlay two color grids, one of images and one of saliency maps
    gradient_grid[gradient_grid < threshold] = 0.0
    gradient_grid = gradient_grid * 255.0  # gradient grid goes from 0 to 1
    example_grid = (example_grid + 1) / 2.0 * 255.0  # example grid goes from -1 to 1
    return cv2.addWeighted(example_grid, 0.5, gradient_grid, 0.5, 0)

