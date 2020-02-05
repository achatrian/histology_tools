import sys

import numpy as np
import torch
import numpy as np

sys.path.insert(0, '../')
import utils
import tile_generator

class MyModel(object):
    '''
    A toy model that randomly inverts the colour channels of the input image.
    Replace this with a proper neural network.
    '''

    def __init__(self, *args, **kwargs):
        print('Replace me with a network')

    def forward(self, input, **kwargs):

        if torch.rand(1) > 0.333:
            input[:,0,:,:] =  input[:,0,:,:].unsqueeze(1)*-1
        if torch.rand(1) > 0.333:
            input[:,1,:,:] =  input[:,1,:,:].unsqueeze(1)*-1
        if torch.rand(1) > 0.333:
            input[:,2,:,:] =  input[:,2,:,:].unsqueeze(1)*-1

        return input

def main():
    # parse options
    opts = utils.anyparser()
    opts.tilesize_base = 1024 if not 'tilesize_base' in opts else opts.tilesize_base
    opts.tilesize_out = 512 if not 'tilesize_out' in opts else opts.tilesize_out
    opts.dataroot = './output.dzi' if not 'dataroot' in opts else opts.dataroot
    opts.output =  './cnn_output.dzi' if not 'output' in opts else opts.output
    opts.gpu =  0 if not 'gpu' in opts else opts.gpu

    # model
    model = MyModel(opts)

    dzi = tile_generator.TileGenerator(opts.dataroot, target=opts.output)
    dzi.clean_target(supress_warning=True)  # Cleans the target directory
    tile_gen = dzi.get_tile(area_thres_percent=0.02, shuffle=False, tilesize_base=(opts.tilesize_base, opts.tilesize_base), tilesize_out=(opts.tilesize_out, opts.tilesize_out), coord_only=True)

    with torch.no_grad():
        for tile_count, (x,y) in enumerate(tile_gen):
            # fn converts image from numpy to torch format, through the pytorch model, then back to numpy
            fn = lambda x: ((model.forward(torch.Tensor(np.moveaxis(x.astype(np.float), (0,1,2), (1,2,0))*2/255-1).unsqueeze(0).cuda(opts.gpu),
                                           opts=opts).cpu().squeeze(0).permute(1,2,0).numpy()+1)/2*255).astype(np.uint8)
            dzi.process_region((x,y), 1, (opts.tilesize_out, opts.tilesize_out), fn, border=0)
            print('Done tile {}; x:{}; y:{}'.format(tile_count, x, y))

    dzi.downsample_pyramid(1)
    dzi.copy_level([0], overwrite=False)

    dzi.auto_crop(padding=1024, border=255)

    dzi_output = tile_generator.TileGenerator(opts.output, target=opts.output, clean_target=False)
    thumb_cropped = dzi_output.get_thumbnail(dzi_output.width / 8)
    utils.multiplot(thumb_cropped, dpi=200)

    print("Finished!")

if __name__ == '__main__':
    main()
