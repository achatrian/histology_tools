'''
Child of DZIIO, with functions to
1) Automatically mask regions with tissue (H&E only, not tested for other stains)
1) Automatically extract tiles with tissue above some % area.
2) Automatically crop dzi images
'''

import numpy as np
import cv2
import warnings
from PIL import Image
from scipy import interpolate
import skimage.morphology as morp
from . import DZIIO


class TileGenerator(DZIIO):
    '''
        inherits openslide class with custom functions
    '''

    def __init__(self, src, target=None, px_size=None, mask_px_size=10, no_mask=False, read_from_base=False, **kwargs):
        '''
        :param src:    Path to the .dzi file
        :param target: Target .dzi file to write to.
        :param px_size: micron per pixel at the pyramid's base level. By default it would try to read the mpp from a json file stored under
                        dzi_files/properties.json if it exists but this could be overwritten.
        :param mask_px_size:    Pixel size of the thumbnail in um used for generating mask where there is tissue.
        :param no_mask: Don't generate mask to save time.
        '''
        super(TileGenerator, self).__init__(src, target=target, **kwargs)

        self.px_size = px_size if px_size is not None else 0.22
        self.mask_px_size = mask_px_size

        self.up_ratio = mask_px_size / self.px_size     # >1
        self.down_ratio = 1 / self.up_ratio             # <1

        self.mask_size = int( max(self.height, self.width) / mask_px_size * self.px_size )
        if not no_mask:
            self.generate_mask()

        if read_from_base:
            self.get_tile = self.get_tile_base
        else:
            self.get_tile = self.get_tile_closest

        # kht.plot.multiplot(self.thumb, self.mask)

    def get_tiles(self, area_thres_percent=0.5, shuffle=False, tilesize_base=(1024,1024), tilesize_out=(256,256), overlap=0, coord_only=False, loop=False, border=255):
        '''
        Function for generating tiles given there is enough tissue in the tile.
        :param area_thres_percent: How much tissue there should be in the tile.
        :param shuffle: whether to shuffle the
        :param tilesize_base: size of the tile at level 0.
        :param tilesize_out:  size of tile to output. Always samples from level 0 for better image quality.
        :param overlap: how many pixels to overlap with adjacent slide on ONE side.
                        eg For tilesize==(1024,1024) and overlap==512 would give you twice as many tiles.
        :param coord_only: returns only (x,y) coordinates but not the actual image.
        :param loop: generator is infinite loop
        :param border: if not None, allows reading regions outside the slide's width/height. Border regions will be greyscale (0-255) given by this parameter.
        :param read_from_base: only downsample from the highest resolution level of the pyramid
        :return: generator that returns a Tile object containing (PIL Image, x, y)
        '''

        assert tilesize_base[0]*tilesize_out[1] == tilesize_base[1]*tilesize_out[0] # Make sure in/out aspect ratio is the same

        notcompleted = True
        list_x = range(np.int(np.floor(self.width / (tilesize_base[0]-overlap))+1))
        list_y = range(np.int(np.floor(self.height / (tilesize_base[1]-overlap))+1))

        tile_mask_width, tile_mask_height = self.slide_to_mask((tilesize_base[0], tilesize_base[1]))
        tile_mask_width = np.maximum(np.int(tile_mask_width), 1)
        tile_mask_height = np.maximum(np.int(tile_mask_height), 1)

        while loop or notcompleted:
            if shuffle:     # Whether to shuffle the slides. If false, yields slides sequentially from x=0,y=0
                list_x = np.random.permutation(list_x)
                list_y = np.random.permutation(list_y)

            for i in list_x:
                for j in list_y:
                    x = i * (tilesize_base[0]-overlap)
                    y = j * (tilesize_base[1]-overlap)
                    mask_coord = self.slide_to_mask((x, y))
                    x_mask = np.int(mask_coord[0])
                    y_mask = np.int(mask_coord[1])

                    # if (x_mask + tile_mask_width < self.mask.shape[1] and y_mask + tile_mask_height < self.mask.shape[0]):
                    if (self.masked_percent(x_mask, y_mask, tile_mask_width, tile_mask_height) > area_thres_percent):
                        if coord_only:
                            yield (x,y)
                        else:
                            tile = self.get_tile((x, y), tilesize_base, tilesize_out, border)
                            yield Tile(tile, x, y)
            notcompleted = False

    def get_tile_base(self, r, tilesize_base, tilesize_out, border):
        tile = Image.fromarray(self.read_region(r, 0, tilesize_base, border=border))
        return tile.resize(tilesize_out)

    def get_tile_closest(self, r, tilesize_base, tilesize_out, border):
        level = np.int(np.log2(np.float(tilesize_base[0]) / np.float(tilesize_out[0])))
        if np.isclose(level, np.log2(np.float(tilesize_base[0]) / np.float(tilesize_out[0]))):
            tile = Image.fromarray(self.read_region(r, level, tilesize_out, border=border))
            return tile
        else:
            closest_level_tilesize = (np.int(tilesize_base[0] / np.power(2, level)), np.int(tilesize_base[1] / np.power(2, level)))
            tile = Image.fromarray(self.read_region(r, level, closest_level_tilesize, border=border))
            return tile.resize(tilesize_out)

    def get_stack(self, r, tilesize, levels, border):
        """
        Returns a list of images same pixel dimensions at different magnifications centred around the same location.
        :param r: Centre location at base coordinate
        :param tilesize:
        :param levels: A list of levels. eg [2, 3, 4]
        :param border:
        :return:
        """

        img = []
        for level in levels:
            _img = self.read_region(r, level, tilesize, border=border, mode=1)
            img.append(_img)

        return img

    def generate_mask(self, method='hsv_otsu', kernelsize=20):
        '''
        Using the a thumbnail of the slide, generates a mask with 1s where tissue is present.
        :return: None
        '''

        if method=="hsv_otsu":
            self.thumb = np.array(self.get_thumbnail(self.mask_size))
            self.mask = cv2.cvtColor(self.thumb, cv2.COLOR_RGB2HSV).astype(float)
            # self.mask = self.mask[:,:,0]*self.mask[:,:,1]*(255-self.mask[:,:,2])
            self.mask = self.mask[:,:,1]*np.minimum(np.maximum(255-self.mask[:,:,2], 128), 10)
            self.mask = np.cast[np.uint8](self.mask/np.max(self.mask)*255)

            kernel = np.ones((kernelsize, kernelsize), np.uint8)

            ret, _ = cv2.threshold(self.mask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            self.mask = cv2.morphologyEx(np.cast[np.uint8](self.mask>ret), cv2.MORPH_CLOSE, kernel)
            self.mask = morp.remove_small_objects(self.mask.astype(bool), min_size=int(10000/self.mask_px_size))
        else:
            print("Please code for other methods for thresholding tissue here.")

    def get_mask(self, x, y, downsample=1):
        '''
         # Returns high resolution mask given the top left coordinates of the tile.
         # downsample is relative to original image
         :param x:   coordinate at level 0
         :param y:   coordinate at level 0
         :param downsample: relative to original image
         :return: mask where pixels with tissue are 1.
         '''

        mask = np.array(self.mask)  # Due to the annoying way how row/col are swapped between PIL and numpy array
        height, width = mask.shape

        old_y, old_x = np.mgrid[0:height, 0:width]
        old_points = np.array([old_x.ravel(), old_y.ravel()]).T

        mask_linear = mask.ravel()

        # Now the query points
        mask_coord = self.slide_to_mask((x, y))
        tile_mask_width, tile_mask_height = self.slide_to_mask((self.tilesize))
        x_mask = np.int(mask_coord[0])
        y_mask = np.int(mask_coord[1])
        tile_mask_width = np.int(tile_mask_width)
        tile_mask_height = np.int(tile_mask_height)
        new_y, new_x = np.mgrid[0:self.tilesize[1]:downsample, 0:self.tilesize[0]:downsample]
        new_x = new_x * tile_mask_width / self.tilesize[0] + x_mask
        new_y = new_y * tile_mask_height / self.tilesize[1] + y_mask

        mask = interpolate.griddata(old_points, mask_linear, (new_x, new_y), method='nearest')
        return mask

    def masked_percent(self, x, y, tile_mask_width, tile_mask_height):
        '''
        Calculates the percent of a tile filled with tissue
        :param x: x coordinate in mask
        :param y: y coordinate in mask
        :param tile_mask_width: width of tile in mask
        :param tile_mask_height: height of tile in mask
        :return: % of tile that is masked as 1
        '''

        area = tile_mask_width * tile_mask_height
        filled = np.array(self.mask)[int(np.maximum(0, y)):int(y + tile_mask_height), int(np.maximum(0, x)):int(x + tile_mask_width)]
        filled = np.sum(filled == 1)
        return filled / area

    def mask_to_slide(self, r, dtype='float', level=0):
        '''
        Functions for mapping coordinate of smaller image to a level in the image pyramid.
        :param r: tuple coordinates (x,y) of thumbnail
        :param dtype: data type
        :param level: level of the image pyramid. 0 for the full image. Supports float.
        :return: integer tuples (x,y) in full image
        '''
        x = np.floor(self.up_ratio * r[0] * np.power(0.5, level))
        y = np.floor(self.up_ratio * r[1] * np.power(0.5, level))

        if dtype=='float':
            return (x, y)
        else:
            return (np.int(x), np.int(y))

    def slide_to_mask(self, r, dtype='float'):
        '''
        Functions for mapping coordinate of full image to thumbnail
        :param r: tuple coordinates (x,y) of full image
        :param dtype: data type
        :return: integer tuples (x,y) in thumbnail
        '''
        x = np.floor(self.down_ratio * r[0])
        y = np.floor(self.down_ratio * r[1])

        if dtype=='float':
            return (x, y)
        else:
            return (np.int(x), np.int(y))

def find_autorotate_angle(dzi):
    """
    # Todo
    Automatically finds the principle angle of the needle biopsy
    :param dzi:
    :return: angle in deg
    """
    dzi.get_mask()

    return 0

# Object to store the tile
class Tile(object):
    def __init__(self, image, x, y):
        self.image = image
        self.x = x
        self.y = y

# -------------- Deprecated Names -------------
def Tile_generator(src, target=None, px_size=None, mask_px_size=10, **kwargs):
    warnings.warn("Tile_generator class has been renamed to TileGenerator", DeprecationWarning)
    return TileGenerator(src, target=target, px_size=px_size, mask_px_size=mask_px_size, **kwargs)
