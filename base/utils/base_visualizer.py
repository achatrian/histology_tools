import os
from pathlib import Path
import sys
import ntpath
import csv
import time
from collections import OrderedDict
from skimage.transform import resize
from . import utils

if sys.version_info[0] == 2:
    VisdomExceptionBase = Exception
else:
    VisdomExceptionBase = ConnectionError


# save images to the disk
def save_images(webpage, visuals, image_path, aspect_ratio=1.0, width=256):
    image_dir = webpage.get_image_dir()
    short_path = ntpath.basename(image_path[0])
    name = os.path.splitext(short_path)[0]

    webpage.add_header(name)
    ims, txts, links = [], [], []

    for label, im_data in visuals.items():
        if not label.endswith('label'):
            im = utils.tensor2im(im_data[0], label.endswith("_map"))  # saving the first images
            image_name = '%s_%s.png' % (name, label)
            save_path = os.path.join(image_dir, image_name)
            h, w, _ = im.shape
            if aspect_ratio > 1.0:
                im = resize(im, (h, int(w * aspect_ratio)), order=2, preserve_range=True)  # TODO test
            if aspect_ratio < 1.0:
                im = resize(im, (int(h / aspect_ratio), w), order=2, preserve_range=True)
            utils.save_image(im, save_path)

            ims.append(image_name)
            txts.append(label)
            links.append(image_name)
    webpage.add_images(ims, txts, links, width=width)


class BaseVisualizer:
    def __init__(self, opt):
        r"""Superclass of task-specific visualizer
        :param opt:
        """
        self.display_id = opt.display_id
        self.use_html = opt.is_train and opt.html
        self.win_size = opt.display_winsize
        self.name = opt.experiment_name
        self.opt = opt
        self.saved = False
        if self.display_id > 0 and not self.opt.no_visdom:
            import visdom
            self.ncols = opt.display_ncols
            self.vis = visdom.Visdom(server=opt.display_server, port=opt.display_port, env=opt.display_env, raise_exceptions=True)

        if self.use_html:
            self.web_dir = os.path.join(opt.checkpoints_dir, self.opt.experiment_name, 'web')
            self.img_dir = os.path.join(self.web_dir, 'images')
            print('create web directory %s...' % self.web_dir)
            Path(self.web_dir).mkdir(exist_ok=True), Path(self.img_dir).mkdir(exist_ok=True)
        self.log_name = os.path.join(opt.checkpoints_dir, self.opt.experiment_name, 'loss_log.txt')
        self.image_size = 256
        with open(self.log_name, "a") as log_file:
            now = time.strftime("%c")
            log_file.write('================ Training Loss (%s) ================\n' % now)

    def reset(self):
        self.saved = False

    def throw_visdom_connection_error(self):
        raise ConnectionError("Could not connect to Visdom server (https://github.com/facebookresearch/visdom) for displaying training progress.\nYou can suppress connection to Visdom using the option --display_id -1. To install visdom, run \n$ pip install visdom\n, and start the server by \n$ python -m visdom.server.\n")

    def log_losses_metrics_to_file(self, losses_and_metrics, epoch, iters):
        write_path = Path(self.opt.checkpoints_dir)/self.opt.experiment_name/'results'
        losses_and_metrics['epoch'] = epoch
        losses_and_metrics['iters'] = iters
        losses_and_metrics.move_to_end('iters', last=False)
        losses_and_metrics.move_to_end('epoch', last=False)
        columns = list(losses_and_metrics.keys())
        with write_path.open('w') as write_file:
            csv_log = csv.DictWriter(write_file, fieldnames=columns, delimiter='\t')
            csv_log.writerow(losses_and_metrics)

    def print_current_losses_metrics(self, epoch, iters, losses, metrics, t=None, t_data=None):
        if type(epoch) is str and epoch.isdigit():
            epoch = int(epoch)
        if iters:  # iter is not given in validation/testing (confusing?)
            message = '(epoch: {:d}, iters: {:d}, time: {:.3f}, data: {:.3f}) '.format(epoch, iters, t, t_data)
        else:
            message = '(epoch: {}, validation) '.format(epoch)
        for i, (k, v) in enumerate((OrderedDict(losses, **metrics)).items()):  # not displayed in correct order in python <3.6
            if not iters:
                k = '_'.join(k.split('_')[0:-1])
            if len(k) > 10:
                k = k[:10]  # shorten very long strings
            message += '{}: {:.3f} '.format(k, v)
        print(message)
        with open(self.log_name, "a") as log_file:
            log_file.write(f'{message}\n')
        if iters:
            self.log_losses_metrics_to_file(OrderedDict(losses, **metrics), epoch, iters)
        return message

    # |visuals|: dictionary of images to display or save
    def display_current_results(self, visuals, visuals_paths, epoch, save_result):
        r""" Abstract method to plot images in visdom as table
        :param visuals:
        :param visuals_paths:
        :param epoch:
        :param save_result:
        :return:
        """
        pass

    # losses: dictionary of error labels and values
    # metrics: same format as |metrics| of plot_current_metrics
    def plot_current_losses_metrics(self, epoch, epoch_progress, losses, metrics):
        r"""Abstract method to plot both metrics and losses
        :param epoch:
        :param epoch_progress:
        :param losses:
        :param metrics:
        :return:
        """
        pass





