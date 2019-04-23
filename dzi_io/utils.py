import argparse

import numpy as np
import torch
import matplotlib.pyplot as plt

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