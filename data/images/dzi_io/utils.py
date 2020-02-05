import argparse
import json
import io
import inspect
from functools import wraps
import warnings

import numpy as np
import torch
import cv2
from matplotlib import pyplot as plt


def anyparser(parser=None, description="Arguments"):
    '''
    Parser that can take in arbitary arguments and converts it into a dict.
    eg. python test.py -x 1 -y thisisastring -z [1,2,a]   ---> returns {'x':1, 'y':'thisisastring', 'z':'[a,1,2]'}
    :param parser:      It is possible to extend from an existing parser.
    :param description: Write something useful
    :return: dictionary containing argument names and values.

    Example:

    test = anyparser("Put in your arguments")
    print("arguments are: {}".format(test))
    '''

    if parser is None:
        parser = argparse.ArgumentParser(description=description)
    # parser.add_argument('-x', required=False, default=None,)
    parsed, unknown = parser.parse_known_args()  # this is an 'internal' method
    # which returns 'parsed', the same as what parse_args() would return
    # and 'unknown', the remainder of that
    # the difference to parse_args() is that it does not exit when it finds redundant arguments

    for arg in unknown:
        if arg.startswith(("-", "--")):
            # you can pass any arguments to add_argument
            parser.add_argument(arg, type=str)

    args = parser.parse_args()

    for i, key in enumerate(args.__dict__):
        if (isinstance(args.__dict__[key], str)):
            try:
                args.__dict__[key] = int(args.__dict__[key])
            except:
                try:
                    args.__dict__[key] = float(args.__dict__[key])
                except:
                    pass

    return args

#Plots multiple subplot. Attempts to save to path savefig if it is not None
#opts should be a list with length==len(args) that will be executed by exec() at every plot.
def multiplot(*args, nrows=0, ncols=0, title=None, figsize=(15,10), savefig=None, opts=None, dpi=100, block=True):

    # Plots torch tensor automatically.
    # If it's a 3D torch.Tensor, axis==0 is colour channel. Swap it to axis==2.
    # If it's a 4D torch.Tensor, axis==1 is colour channel. axis==0 is batch. Plot only the 1st batch.
    def convert_torch(arg):
        try:
            if isinstance(arg, torch.Tensor):
                if len(arg.shape)==3:
                    if arg.shape[0]!=3:
                        arg = arg[0,:,:]
                    else:
                        arg = torch.squeeze(arg.permute(1, 2, 0))

                elif len(arg.shape)==4:
                    if arg.shape[1]!=3:
                        arg = arg[0,0,:,:]
                    else:
                        arg = torch.squeeze(arg[0,:,:,:]).permute(1,2,0)
        except:
            pass

        return arg

    if len(args)==1:
        if type(args[0])==tuple: # If it is a tuple, treat each item as an argument
            args = args[0]
            nplots = len(args)
        else:
            nplots = 1
    else:
        nplots = len(args)

    # Want a number of subplots depending on nplot: 1, 1x2, 2x2, 2x2, 2x3, 2x3, 3x3, 3x3, 3x3, 3x4, 3x4, 3x4, 4x4 ...
    if nrows==0 and ncols==0:
        nrows = np.int(np.ceil(0.5*(np.sqrt(1+4*nplots)-1)))
        ncols = np.int(np.ceil(np.sqrt(nplots)))
    elif nrows==0:
        nrows = nplots
        ncols = 1
    elif ncols==0:
        ncols = nplots
        nrows = 1

    if savefig is None:
        plt.ioff()

    if nplots == 1:
        im = plt.imshow(convert_torch(args[0]))
        if opts is not None: exec(opts[0])
        if title is not None: plt.title(title)
    elif np.minimum(nrows, ncols) == 1:
        f, axarr = plt.subplots(nrows, ncols, figsize=figsize)
        for i in range(nplots):
            im = axarr[i].imshow(convert_torch(args[i]))
            if opts is not None: exec(opts[i])
            if title is not None:
                if title[i] is not None: axarr[i].set_title(title[i])
    else:
        f, axarr = plt.subplots(nrows, ncols, figsize=figsize)
        n = 0
        for i in range(nrows):
            for j in range(ncols):
                if n < nplots:
                    im = axarr[i,j].imshow(convert_torch(args[n]))
                    if opts is not None: exec(opts[n])
                    if title is not None:
                        if title[n] is not None: axarr[i,j].set_title(title[n])
                    n = n + 1

    plt.axis('off')

    if (savefig is not None):
        try:
            plt.savefig(savefig, bbox_inches='tight', dpi=dpi)
            plt.close()
            return 0
        except Exception as e:
            print('Cannot save figure.', e)

    plt.show(block=block)

# Saves and loads dict to/from json (depending on whether data is None)
def json_io(filename, data=None, sort_keys=True):
    '''
    :param filename: filename to save to.
    :param data: dict to be saved as json
    :param sort_keys:   bool, whether to sort the keys in the json file.
                        Only works if order of all keys can be compared.
    '''
    try:
        to_unicode = unicode
    except NameError:
        to_unicode = str

    if data is None:
        # Read JSON file
        with open(filename) as data_file:
            data = json.load(data_file)
        return data
    else:
        # Write JSON file
        with io.open(filename, 'w', encoding='utf8') as outfile:
            str_ = json.dumps(data,
                              indent=4, sort_keys=sort_keys, cls=NumpyEncoder,
                              separators=(',', ': '), ensure_ascii=False)
            outfile.write(to_unicode(str_))
        return 0


class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
            np.int16, np.int32, np.int64, np.uint8,
            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32,
            np.float64)):
            return float(obj)
        elif isinstance(obj,(np.ndarray,)): #### This is the fix
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def rotate_bound(image, angle):
    """
    # Rotation with padding
    # https://www.pyimagesearch.com/2017/01/02/rotate-images-correctly-with-opencv-and-python/
    # grab the dimensions of the image and then determine the center
    """
    (h, w) = image.shape[:2]
    (cX, cY) = (w / 2, h / 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (nW, nH))


def deprecated(reason):
    """
    This is a decorator which can be used to mark functions
    as deprecated. It will result in a warning being emitted
    when the function is used.
    """

    if isinstance(reason, (type(b''), type(u''))):

        # The @deprecated is used with a 'reason'.
        #
        # .. code-block:: python
        #
        #    @deprecated("please, use another function")
        #    def old_function(x, y):
        #      pass

        def decorator(func1):

            if inspect.isclass(func1):
                fmt1 = "Call to deprecated class {name} ({reason})."
            else:
                fmt1 = "Call to deprecated function {name} ({reason})."

            @wraps(func1)
            def new_func1(*args, **kwargs):
                warnings.simplefilter('always', DeprecationWarning)
                warnings.warn(
                    fmt1.format(name=func1.__name__, reason=reason),
                    category=DeprecationWarning,
                    stacklevel=2
                )
                warnings.simplefilter('default', DeprecationWarning)
                return func1(*args, **kwargs)

            return new_func1

        return decorator

    elif inspect.isclass(reason) or inspect.isfunction(reason):

        # The @deprecated is used without any 'reason'.
        #
        # .. code-block:: python
        #
        #    @deprecated
        #    def old_function(x, y):
        #      pass

        func2 = reason

        if inspect.isclass(func2):
            fmt2 = "Call to deprecated class {name}."
        else:
            fmt2 = "Call to deprecated function {name}."

        @wraps(func2)
        def new_func2(*args, **kwargs):
            warnings.simplefilter('always', DeprecationWarning)
            warnings.warn(
                fmt2.format(name=func2.__name__),
                category=DeprecationWarning,
                stacklevel=2
            )
            warnings.simplefilter('default', DeprecationWarning)
            return func2(*args, **kwargs)

        return new_func2

    else:
        raise TypeError(repr(type(reason)))
