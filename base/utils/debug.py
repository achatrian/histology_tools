from pathlib import Path
import datetime
from matplotlib import pyplot as plt
import imageio


def show_image(image):
    # plot images quickly
    plt.imshow(image)
    plt.show()


def save_image(image, path=Path('/well/rittscher/users/achatrian/debug'), name=None):
    # save images when plotting in pycharm remotely doesn't work
    try:
        if not name:
            path = path / (str(datetime.datetime.now()) + '.png')  # add date to make unique
        else:
            path = path / (name + '.png')
        imageio.imsave(path, image)
    except NotADirectoryError:
        path = Path('/mnt/rittscher/users/achatrian/debug')
        if not name:
            path = path / (str(datetime.datetime.now()) + '.png')  # add date to make unique
        else:
            path = path / (name + '.png')
        imageio.imsave(path, image)


