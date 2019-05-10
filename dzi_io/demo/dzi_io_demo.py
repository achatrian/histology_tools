# For testing IO from DZI files
import os, sys, shutil
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, '../')
from dzi_io import *
import utils

def main():

    parser = utils.anyparser()
    if 'dzi_src' in parser:
        dzi_src = parser.dzi_src
    else:
        dzi_src = './example.dzi'
    if 'dzi_target' in parser:
        dzi_target = parser.dzi_target
    else:
        dzi_target = './output.dzi'
    fn = lambda x: norm_image(x, max_norm=200)

    dzi = DZI_IO(dzi_src, target=dzi_target, clean_target=True)

    # test that we are calculating the dimension from the tile correctly as read from the metadata
    if True:
        dzi.wh.pop(0)
        dzi.level_dimensions(0)
        assert((dzi.width, dzi.height)==dzi.level_dimensions(0))

    # Now try processing all the tiles that contains tissues
    if True:
        thumb = dzi.get_thumbnail(dzi.width/32)
        plt.figure(dpi=300)
        plt.imshow(thumb)
        plt.show()

    # Now try read_region()
    x0, y0 = int(dzi.width/2), int(dzi.height/2)
    if True:
        img00 = dzi.read_region((0, 0), 1, (2048,2048), mode=1, border=0)    # Should return image with black borders.
        img01 = dzi.read_region((dzi.width, dzi.height), 1, (2048, 2048), mode=1, border=255)  # Should return image with white borders.
        img1 = dzi.read_region((x0, y0), 1, (512, 512), mode=1)
        img2 = dzi.read_region((x0, y0), 2, (512, 512), mode=1)
        img3 = dzi.read_region((x0, y0), 3, (512, 512), mode=1)
        img4 = dzi.read_region((x0, y0), 4, (512, 512), mode=1)
        utils.multiplot(img00, img01, img1, img2, img3, img4, dpi=200)

    # Now try processing the image
    test_level = 2
    if True:
        try:
            dzi.clean_target(supress_warning=True)    # Cleans the target directory
        except:
            pass
        img = dzi.process_region((x0, y0), test_level, (512, 512), fn, mode=1, read_target=True)
        img = dzi.process_region((x0 + 1024, y0 + 256), test_level, (256, 256), lambda x: np.flip(x, axis=2), mode=1, read_target=True)
        img = dzi.process_region((x0 - 1024, y0 - 256), test_level, (256, 256), lambda x: np.roll(x, 1, axis=2), mode=1, read_target=True)
        # dzi.copy_level([2,3,4,5,6])
        dzi.update_pyramid(test_level, level_end=None)

    # Try rotating the image pyramid
    if True:
        dzi = DZI_IO(dzi_target, target=dzi_target, clean_target=False)
        dzi.rotate(20)
        dzi_output = DZI_IO(dzi_target, target=dzi_target, clean_target=False)
        thumb_rot = dzi_output.get_thumbnail(dzi_output.width / 32)
        utils.multiplot(thumb_rot)

    # Try cropping the image pyramid
    if True:
        r1 = (int(dzi.width * 0.05), int(dzi.height * 0.35))
        r2 = (int(dzi.width * 0.90), int(dzi.height * 0.65))
        dzi.crop(r1, r2, t=254, v=16)
        dzi.close()
        dzi_output = DZI_IO(dzi_target, target=dzi_target, clean_target=False)
        thumb_cropped = dzi_output.get_thumbnail(dzi_output.width / 32)
        utils.multiplot(thumb_cropped)

    # Try adding two dzi images. Note that the inputs can be different resolution.
    # This will be useful in cases where a model is used to process one level and we want
    if True:

        b4_addition = os.path.join(os.path.split(dzi_target)[0], 'b4_addition')
        os.remove(b4_addition + '.dzi') if os.path.exists(b4_addition + '.dzi') else None
        shutil.rmtree(b4_addition + '_files') if os.path.exists(b4_addition + '_files') else None
        shutil.move(os.path.splitext(dzi_target)[0] + '_files', b4_addition + '_files')
        shutil.move(dzi_target, b4_addition + '.dzi')
        dzi.close()         # Somehow relying on the destructor is not very reliable it's better to call close() explicitly.
        dzi = DZI_IO(dzi_src, target=dzi_target)
        dzi2 = DZI_IO(b4_addition + '.dzi', target=dzi_target)

        seq = DZI_Sequential((dzi, dzi2), lambda x, y: (x + y)/2)
        seq.evaluate()

    print('Finished')

# Define some toy function for processing the image
def norm_image(img, max_norm=1.0):
    is_PIL = False

    if isinstance(img, Image.Image):
        is_PIL = True
        img = np.array(img)

    img = (img - np.min(img)) / (np.max(img) - np.min(img)) * max_norm

    if is_PIL:
        img = Image.fromarray(img)

    return img

if __name__ == "__main__":
    main()