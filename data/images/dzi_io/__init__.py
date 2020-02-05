import os
import re
import shutil
import warnings
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from . import utils


class DZIIO(object):

    def __init__(self, src, target=None, clean_target=False):
        '''

        :param src: Name of .dzi file to read
        :param target: Name of the .dzi file to write to
        :param clean_target: Delete everything in the target directory to start with a clean dir.
        '''
        self.src = src
        self.target = target
        self.src_dir = str(src).replace('.dzi', '') + "_files"
        self.target_dir = str(target).replace('.dzi', '') + "_files" if target is not None else None

        self.max_colrow = {}            # self.max_colrow[level] = [max_col, max_row]  maximum number of col and row in each level.
        self.wh = {}                    # self.wh[level] = (width, height) --- width and height of the pyramid level.
        self.end_wh = {}                # self.wh[level] = (width, height) --- width and height of last tile in the pyramid level.
        self.mpp = None                 # microns per pixel
        self.cropped = False            # Code should complain if you try to process images after cropping the pyramid.
        self.resized = False

        # Read source .dzi for metadata
        f = open(src, 'r')
        meta = f.read()
        self.format = re.search(r'Image Format="(\w{3,4})"', meta).group(1)
        self.overlap = int(re.search(r'Overlap="(\d{1,4})"', meta).group(1))
        self.tilesize = int(re.search(r'TileSize="(\d{1,5})"', meta).group(1))
        self.height = int(re.search(r'Height="(\d{1,6})"', meta).group(1))
        self.width = int(re.search(r'Width="(\d{1,6})"', meta).group(1))
        f.close()

        if os.path.exists(os.path.join(self.src_dir, 'properties.json')):
            self.properties = utils.json_io(os.path.join(self.src_dir, 'properties.json'))
            os.makedirs(self.target_dir) if (self.target is not None) and (not os.path.exists(self.target_dir)) else None
            utils.json_io(os.path.join(self.target_dir, 'properties.json'), self.properties) if (self.target is not None) else None
            self.mpp = self.properties['mpp'] if 'mpp' in self.properties else self.mpp     # Try using this key first as mpp.
            self.mpp = self.properties['openslide.mpp-x'] if ('openslide.mpp-x' in self.properties and self.mpp is None) else self.mpp
            self.mpp = self.properties['openslide.mpp-y'] if ('openslide.mpp-y' in self.properties and self.mpp is None) else self.mpp

        self.level_count = len([x for x in os.listdir(self.src_dir) if os.path.isdir(os.path.join(self.src_dir, x))])
        try:
            assert(self.level_dimensions(0)==(self.width, self.height))
        except AssertionError:
            print("Warning! Width,Height from {} is ({}, {}) from the dzi's file. However from the tiles it was evaluated to be ({}, {}). \
            May have issues later. Using value evaluated from the tiles.".format(self.src.name, self.width, self.height, self.level_dimensions(0)[0], self.level_dimensions(0)[1]))
            self.width, self.height = self.level_dimensions(0)
        assert(set([int(x) for x in [x for x in os.listdir(self.src_dir) if os.path.isdir(os.path.join(self.src_dir, x))]]) == set(range(self.level_count))) # Make sure dir numberings are contagious from 0 to "self.level_count-1" if there is >1 folder

        # Copy the .dzi file and create the target directory
        if target is not None:
            # Create the target directory
            if not os.path.exists(self.target_dir):
                os.makedirs(self.target_dir)

            # Copy .dzi to target
            shutil.copyfile(src, target) if src!=target else None

        if clean_target and (self.target!=self.src) and (self.target is not None):
            self.clean_target(supress_warning=True)

    def __del__(self):
        self.close()

    def close(self):
        '''
        This should re-evaluate the pyramid in the target directory and update the .dzi accordingly.
        :return:
        '''

        if (self.target is None) or (not os.path.exists(self.target_dir)) \
                or len([x for x in os.listdir(self.target_dir) if os.path.isdir(os.path.join(self.target_dir, x))])==0:
            return

        levels = [int(x) for x in os.listdir(self.target_dir) if os.path.isdir(os.path.join(self.src_dir, x)) or os.path.isdir(os.path.join(self.target_dir, x))]
        levels = sorted(levels)

        try:
            width_new, height_new = self.level_dimensions(self.target_level_count - levels[-1] - 1, src='target')
        except:
            width_new, height_new = self.level_dimensions(self.level_count - levels[-1] - 1, src='target')

        # Update the dzi file.
        f = open(self.target, 'r')
        meta = f.read()
        f.close()
        meta = re.sub(r'Height="(\d{1,6}")', 'Height="{}"'.format(height_new), meta)
        meta = re.sub(r'Width="(\d{1,6}")', 'Width="{}"'.format(width_new), meta)
        f.close()
        with open(self.target, 'w') as f:
            f.write(meta)  # writing to target .dzi file after creating new DZIIO object as the constructor itself copies the dzi file.

        # Update the mpp properties json file. If we crop the no. of levels might change but mpp wouldn't change.
        if (not self.resized) and not (self.cropped) and (self.mpp is not None):
            ratio = self.width / width_new
            mpp_new = np.float(self.mpp) * ratio
            self.properties['mpp'] = mpp_new
            utils.json_io(os.path.join(self.target_dir, 'properties.json'), self.properties)

        elif self.resized:
            pass

        else:
            self.properties['mpp'] = self.mpp
            utils.json_io(os.path.join(self.target_dir, 'properties.json'), self.properties)

    # ------ Methods for reading images, meant to work similar to openslide_python with equivalent names------
    def read_region(self, location, level, size, mode=0, src='src', border=None):
        '''
        :param location: (x,y) in the level 0 reference frame (highest magnification) unless specified otherwise in "mode"
        :param level:   Level to read the tiles from
        :param size: (width, height) from the level specified
        :param mode: 0: location (x,y) specifies the top left corner; 1: location (x,y) specifies the centre of the image.
                     0: location specified in level 0 reference frame; 2: location specified in the given level.
                     Mode is given by the sum of the above.
        :param src: whether to read the tiles from only the source (src) or target (target). If set to 'target', will try to read from the self.target_dir first.
        :param border: if not None, allows reading regions outside the slide's width/height. Border regions will be greyscale (0-255) given by this parameter.
                        Alternative, set it to "max"/"min" to pad the image with the maximum/minimum of the image
        :return:
        '''

        assert(not self.cropped and not self.resized)

        # If mode==1, convert location into top right left corner
        size_base = self.map_xy(size, level_src=level)  # Tile size at the base level
        if mode & 2:
            location = self.map_xy(location, level_src=level)
        if mode & 1:
            location = (location[0] - int(size_base[0]/2), location[1] - int(size_base[1]/2))

        pad_start = [0, 0]
        pad_end = [0, 0]

        if (border is not None) and (isinstance(border, int) or "min" in border or "max" in border):     # Pad areas outside slide as grey
            _r = self.map_xy(location, level_src=0, level_target=level)
            pad_start = [np.maximum(0, -_r[0]), np.maximum(0, -_r[1])]
            pad_end = [np.maximum(0, (_r[0]+size[0])-self.level_dimensions(level)[0]), np.maximum(0, (_r[1]+size[1])-self.level_dimensions(level)[1])]
        else:
            try:
                assert (location[0] >= 0 and location[1] >= 0 and location[0] + size_base[0] < self.width and location[1] + size_base[1] < self.height)
            except Exception as e:
                print("Please ensure that cropped region is within the image or set border = integer to set a border")
                raise e

        tilelist_toplft = self.map_xy_tile(location, 0, level)
        tilelist_btnrgt = self.map_xy_tile((location[0]+size_base[0], location[1]+size_base[1]), 0, level)

        # Now we want the idx of the "bottom right" most tile of the top left and the "top left" most tile of the bottom right to minimise the no. of images loaded
        _tilenames_toplft, idxs_toplft, toplft_coord = tilelist_toplft[-1]
        _tilenames_btnrgt, idxs_btnrgt, btnright_coord = tilelist_btnrgt[0]

        img = self.join_tiles(level, tile_idx=[(idxs_toplft[0], idxs_btnrgt[0]),(idxs_toplft[1], idxs_btnrgt[1])], src=src)

        img = img[toplft_coord[1]:toplft_coord[1]+size[1]-pad_start[1]-pad_end[1],toplft_coord[0]:toplft_coord[0]+size[0]-pad_start[0]-pad_end[0]]

        if any(pad_start) or any(pad_end):
            _img = np.ones((size[1], size[0], 3), dtype=np.uint8)
            if (not isinstance(border, int)) and "max" in border:
                for i in range(3):
                    _img[:,:,i] *= np.max(img[:,:,i])
            elif (not isinstance(border, int)) and "min" in border:
                for i in range(3):
                    _img[:,:,i] *= np.min(img[:,:,i])
            else:
                _img *= border

            if (_img[pad_start[1]:size[1]-pad_end[1], pad_start[0]:size[0]-pad_end[0]].size == 0) or (img.size == 0):
                return _img
            try:
                _img[pad_start[1]:size[1]-pad_end[1], pad_start[0]:size[0]-pad_end[0]] = img
            except:
                pass
            return _img

        return img

    def get_thumbnail(self, size, src='src'):
        '''
        Returns a thumbnail that fits within (size,size).
        Get the image from the next higher zoom level and then rescales image to (<=size, <=size)
        :param size:
        :param src: whether to read the tiles from only the source (src) or target (target). If set to 'target', will try to read from the self.target_dir first.
        :return: thumbnail
        '''

        maxdim = max(self.height, self.width)
        thumb_width, thumb_height = (np.int(self.width/self.height*size),np.int(size)) if self.height > self.width else (np.int(size),np.int(self.height/self.width*size))
        downscale = maxdim / size

        sample_level = np.int(np.floor(np.log2(downscale))) # Get the tiles in this level to produce the thumbnail

        img = self.join_tiles(sample_level, src=src)

        img = Image.fromarray(img).resize((thumb_width, thumb_height))

        return img

    def level_dimensions(self, level, src='src'):
        '''
        level_dimensions(k) are the dimensions of level k.
        :param level:
        :param src: Whether to calculate the width/height from src or target.
        :return:
        '''

        if src == 'src':
            if level in self.wh:
                return self.wh[level]

            level_dir = os.path.join(self.src_dir, "{}".format(self.level_count - level - 1))  # Note that dzi orders dir's magnification in ascending order
        else:
            try:
                level_dir = os.path.join(self.target_dir, "{}".format(self.target_level_count - level - 1))  # Note that dzi orders dir's magnification in ascending order
            except:
                level_dir = os.path.join(self.target_dir, "{}".format(self.level_count - level - 1))  # Note that dzi orders dir's magnification in ascending order

        max_col, max_row = self.get_max_colrow(level, src=src)

        endtile = np.array(Image.open(os.path.join(level_dir, "{:d}_{:d}.{:s}".format(max_col, max_row, self.format))))
        end_height, end_width, _ = endtile.shape

        width = self.calc_length(0, max_col, max_col, end_width)
        height = self.calc_length(0, max_row, max_row, end_height)

        self.wh[level] = (width, height)

        return (width, height)

    # ----- Methods for mapping coordinates between different levels ------
    def join_tiles(self, level, tile_idx=None, src='src'):
        '''
        Joins tiles together as specified by tile_idx and returns the merged image
        :param level:
        :param tile_idx: [(start_column, end_column), (start_row, end_row)]; if "None", join all tiles in the directory
        :param src: whether to read the tiles from only the source (src) or target (target). If set to 'target', will try to read from the self.target_dir first.
        :return: img
        '''

        if src == 'target':
            level_dir_target = os.path.join(self.target_dir, "{}".format(self.level_count - level - 1))

        level_dir = os.path.join(self.src_dir, "{}".format(self.level_count - level - 1))  # Note that dzi orders dir's magnification in ascending order

        max_col, max_row = self.get_max_colrow(level)

        if tile_idx is None:
            tile_idx = [(0, max_col), (0, max_row)]

        tile_idx = (sorted(tile_idx[0]), sorted(tile_idx[1]))

        end_width, end_height = self.get_end_wh(level)

        max_width = self.calc_length(tile_idx[0][0], tile_idx[0][1], max_col, end_width)
        max_height = self.calc_length(tile_idx[1][0], tile_idx[1][1], max_row, end_height)

        img = np.zeros((max_height, max_width, 3), dtype=np.uint8)

        for i in range(tile_idx[0][0], tile_idx[0][1]+1):
            for j in range(tile_idx[1][0], tile_idx[1][1] + 1):

                try:
                    tile = np.array(Image.open(os.path.join(level_dir_target, "{:d}_{:d}.{:s}".format(i, j, self.format))))
                except (UnboundLocalError, FileNotFoundError):
                    tile = np.array(Image.open(os.path.join(level_dir, "{:d}_{:d}.{:s}".format(i, j, self.format))))

                start_y = (j - tile_idx[1][0])*self.tilesize - self.overlap*(j!=0) + self.overlap*(tile_idx[1][0]!=0)
                start_x = (i - tile_idx[0][0])*self.tilesize - self.overlap*(i!=0) + self.overlap*(tile_idx[0][0]!=0)
                end_y_img_pad = self.overlap + self.overlap*(j!=0)
                end_x_img_pad = self.overlap + self.overlap*(i!=0)
                if start_y + self.tilesize + self.overlap > max_height:
                    end_y = max_height
                    end_y_img_pad = 0
                else:
                    end_y = start_y + self.tilesize
                if start_x + self.tilesize + self.overlap > max_width:
                    end_x_img_pad = 0
                    end_x = max_width
                else:
                    end_x = start_x + self.tilesize

                try:
                    img[start_y:end_y+end_y_img_pad, start_x:end_x+end_x_img_pad] = tile[0:self.tilesize+2*self.overlap, 0:self.tilesize+2*self.overlap]
                except ValueError:  # For some reasons sometimes the tiles' dimensions don't quite match due to rounding
                    _img_y, _img_x, _ = img[start_y:end_y+end_y_img_pad, start_x:end_x+end_x_img_pad].shape
                    _tile_y, _tile_x, _ = tile[0:self.tilesize + 2 * self.overlap, 0:self.tilesize + 2 * self.overlap].shape
                    _end_y, _end_x, _ = img[start_y:end_y + end_y_img_pad - (_img_y > _tile_y), start_x:end_x + end_x_img_pad - (_img_x > _tile_x)].shape
                    img[start_y:end_y + end_y_img_pad - (_img_y > _tile_y), start_x:end_x + end_x_img_pad - (_img_x > _tile_x)] = tile[0:_end_y, 0:_end_x]

                # tile[0,:] = 0; tile[:,0] = 0; tile[-1,:] = 0; tile[:,-1] = 0
                # img[start_y:end_y + end_y_img_pad, start_x:end_x + end_x_img_pad] = (0.5*tile[0:self.tilesize + self.overlap, 0:self.tilesize + self.overlap]
                #                                                                      + 0.5*img[start_y:end_y + end_y_img_pad, start_x:end_x + end_x_img_pad])

        return img

    def map_xy(self, r, level_src=0, level_target=0, src_size=None, target_size=None, dtype='int'):
        '''
        Function for mapping coordinates from a source level to a target level in the image pyramid.
        :param r: tuple coordinates (x,y) of src
        :param level_src: level of the image pyramid for input r.
        :param level_target: level of the image pyramid for the output. 0 for maximum zoom.
        :param src_size: If not None, use "src_size" and "target_size" to calculate the ratio instead.
        :param target_size:
        :param dtype: data type
        :return: integer tuples (x,y) in target level
        '''

        if src_size is None or target_size is None:
            ratio = np.power(2.0, level_src - level_target)
        else:
            ratio = target_size/src_size

        x = ratio * r[0]
        y = ratio * r[1]

        if dtype=='int':
            return (np.int(x), np.int(y))
        else:
            return (x, y)

    def map_xy_tile(self, r, level_src=0, level_target=0):
        '''
        Function mapping a coordinate from a level to the tilenames that contain that coordinate and the local coordinate in each tile.
        If self.overlap > 0 may return more than one tilename.

        :param r: (x,y) location in level_src
        :param level_src:   level of which coordinates of r is based
        :param level_target:
        :return: tile_list - a list where each element is a tuple (tilename, "(i,j) col and row index of the tile", "(x,y) in the tile's coordinate")
        '''

        r = [np.maximum(0, np.minimum(r[0], self.level_dimensions(level_src)[0])), np.maximum(0, np.minimum(r[1], self.level_dimensions(level_src)[1]))]

        max_col, max_row = self.get_max_colrow(level_target)
        max_width, max_height = self.level_dimensions(level_target)
        r_target = self.map_xy(r, level_src, level_target)
        r_target = (np.minimum(r_target[0], max_width), np.minimum(r_target[1], max_height))
        tile_list = []

        # Iterate through the tiles and find the top left and bottom right coords
        for i in range(max_col+1):
            if (i*self.tilesize-self.overlap <= r_target[0]) and (r_target[0] <= (i+1)*self.tilesize+self.overlap):
                for j in range(max_row+1):
                    if (j*self.tilesize-self.overlap <= r_target[1]) and (r_target[1] <= (j+1)*self.tilesize+self.overlap):
                        r_tile = (r_target[0] - (i*self.tilesize - self.overlap*(i>0)), r_target[1] - (j*self.tilesize - self.overlap*(j>0)))
                        tile_list.append(["{:d}_{:d}.{:s}".format(i, j, self.format), (i, j), r_tile])

        return tile_list

    def tile_idx2xy(self, idx, level_src=None, level_target=None):
        '''
        Given a tile with index (i,j) at the specified level, returns the coordinates of the top left corner of that tile.
        :param idx: (i,j)
        :param level_src: if not None, (x,y) will be mapped from this level to the level_target reference frame
        :param level_target:
        :return: (x,y)
        '''
        x = idx[0] * self.tilesize - self.overlap * (idx[0] > 0)
        y = idx[1] * self.tilesize - self.overlap * (idx[1] > 0)

        if (level_src is not None) and (level_target is not None):
            x, y = self.map_xy((x,y), level_src, level_target)

        return (x,y)

    # ------ Misc ------
    def get_max_colrow(self, level, src='src'):
        '''
        Gets the indicies of the last column and the last row in a given level.
        :param level:
        :return: (max_col, max_row)
        '''

        if src=='src':

            if level not in self.max_colrow:

                max_col = 0
                max_row = 0

                level_dir = os.path.join(self.src_dir, "{}".format(self.level_count-level-1)) # Note that dzi orders dir's magnification in ascending order
                tile_names = os.listdir(level_dir)

                for tile_name in tile_names:
                    m = re.search(r'(\d{1,3})_(\d{1,3})', tile_name)
                    max_col = np.maximum(max_col, int(m.group(1)))
                    max_row = np.maximum(max_row, int(m.group(2)))

                self.max_colrow[level] = (max_col, max_row)

            return self.max_colrow[level]

        else:

            max_col = 0
            max_row = 0

            try:
                level_dir = os.path.join(self.target_dir, "{}".format(self.target_level_count - level - 1))  # Note that dzi orders dir's magnification in ascending order
            except:
                level_dir = os.path.join(self.target_dir, "{}".format(self.level_count - level - 1))  # Note that dzi orders dir's magnification in ascending order

            tile_names = os.listdir(level_dir)

            for tile_name in tile_names:
                m = re.search(r'(\d{1,3})_(\d{1,3})', tile_name)
                max_col = np.maximum(max_col, int(m.group(1)))
                max_row = np.maximum(max_row, int(m.group(2)))

            return (max_col, max_row)

    def get_end_wh(self, level):
        '''
        Gets the width and height of the bottom right tile in a level.
        :param level:
        :return:
        '''

        if level not in self.end_wh:
            level_dir = os.path.join(self.src_dir, "{}".format(self.level_count - level - 1))  # Note that dzi orders dir's magnification in ascending order

            max_col, max_row = self.get_max_colrow(level)

            endtile = np.array(Image.open(os.path.join(level_dir, "{:d}_{:d}.{:s}".format(max_col, max_row, self.format))))
            end_height, end_width, _ = endtile.shape

            self.end_wh[level] = (end_width, end_height)

        return self.end_wh[level]

    def calc_length(self, start_idx, end_idx, max_n, end_length):
        '''
        Calculates length of several tiles joined together

        :param start_idx: index of the first tile to be joined (0<=start_idx<=end_idx)
        :param end_idx: index of the last tile to be joined (start_idx<=end_idx<=max_n)
        :param max_n: number of tiles in whole level along the dimension of interest
        :param end_length: length of the last tile
        :return: length of the joined image
        '''

        if max_n == 0:
            return end_length

        length = 0

        if end_idx == max_n:
            length += end_length - 2 * self.overlap

        if (start_idx == 0):
            length += self.tilesize + self.overlap
        else:
            length += 2 * self.overlap

        num_mid = end_idx - start_idx + 1 - (end_idx == max_n) - (start_idx == 0)
        length += num_mid * self.tilesize

        if (end_length < 2*self.overlap) and (end_idx == max_n-1):
            length -= 2*self.overlap - end_length

        return length

    def halfsize(self, img):
        '''
        Returns fn that returns img when called.
        :param img:
        :return:
        '''

        if not isinstance(img, Image.Image):
            img = Image.fromarray(img)
        img = img.resize((np.int(img.size[0]/2), np.int(img.size[1]/2)))
        img = np.array(img)

        def call(tile=None):
            if tile is not None:
                return img[0:tile.shape[0], 0:tile.shape[1], 0:tile.shape[2]]
            else:
                return img

        return call

    # ------ Methods for writing into dzi tiles ---------

    def process_region(self, location, level, size, fn, mode=0, read_target=True, border=None):
        '''
        Processes read_region() at a given level. Saves the output in the target directory.
        1) read_region from a level. Will try to read the tiles from target_dir first before searching from source dir.
            - (may involve reading across several .jpeg files)
        2) processed_img = fn(read_region())
        3) Saves output to target_dir's base level.
            - Since read_region may not be aligned
            - if the tile file already exists in the target_dir,
              read the tile file in target_dir

        :param location: function for processing a tile at the base level
        :param level:
        :param size:
        :param fn: function to process read_region(location, level, size)
        :param mode: whether location specified is the centre or top-left of the region.
        :param read_target: should read_region() always try to load tiles saved in target_dir first?
        :param border: if not None, allows reading regions outside the slide's width/height. Border regions will be greyscale (0-255) given by this parameter.
        :return: processed region
        '''

        assert(not self.cropped and not self.resized)
        assert(self.target_dir is not None)

        if read_target:
            src = 'target'
            level_dir_target = os.path.join(self.target_dir, "{}".format(self.level_count - level - 1))
        else:
            src = 'src'

        level_dir = os.path.join(self.src_dir, "{}".format(self.level_count - level - 1))

        # If mode==1, convert location into top right left corner
        size_base = self.map_xy(size, level_src=level)  # Tile size at the base level
        if mode==1:
            location = (location[0] - int(size_base[0]/2), location[1] - int(size_base[1]/2))

        tile = self.read_region(location, level, size, src=src, border=border)

        assert(tile.size!=0)

        processed_img = fn(tile)

        # Now find all the tiles that overlaps with processed_img
        x_start, y_start = self.map_xy(location, level_src=0, level_target=level)       # Top left coord of the processed region in a level
        tilelist_toplft = self.map_xy_tile(location, 0, level)
        tilelist_btnrgt = self.map_xy_tile((location[0]+size_base[0], location[1]+size_base[1]), 0, level)

        for i in range(tilelist_toplft[0][1][0], tilelist_btnrgt[-1][1][0]+1):
            for j in range(tilelist_toplft[0][1][1], tilelist_btnrgt[-1][1][1]+1):
                _x, _y = self.tile_idx2xy((i,j), level_src=level)   # Top left coord of a tile in a level

                try:
                    tile = np.array(Image.open(os.path.join(level_dir_target, "{:d}_{:d}.{:s}".format(i, j, self.format))))
                except (UnboundLocalError, FileNotFoundError):
                    tile = np.array(Image.open(os.path.join(level_dir, "{:d}_{:d}.{:s}".format(i, j, self.format))))

                x_start_tile = np.maximum(0, x_start-_x)
                x_start_region = np.maximum(0, _x-x_start)
                x_end_region = size[0] - ((x_start + size[0]) - (_x + tile.shape[1]))

                y_start_tile = np.maximum(0, y_start-_y)
                y_start_region = np.maximum(0, _y-y_start)
                y_end_region = size[1] - ((y_start + size[1]) - (_y + tile.shape[0]))

                _processed_img = processed_img[y_start_region:y_end_region, x_start_region:x_end_region]
                x_end_tile = _processed_img.shape[1] + x_start_tile
                y_end_tile = _processed_img.shape[0] + y_start_tile

                tile[y_start_tile:y_end_tile, x_start_tile:x_end_tile] = _processed_img

                # Now save the tiles
                if not os.path.exists(os.path.join(self.target_dir, "{}".format(self.level_count - level - 1))):
                    os.makedirs(os.path.join(self.target_dir, "{}".format(self.level_count - level - 1)))
                save_path = os.path.join(self.target_dir, "{}/{:d}_{:d}.{:s}".format(self.level_count - level - 1, i, j, self.format))
                if os.path.exists(save_path):
                    os.remove(save_path)        # Somehow PIL image saves into the symlink if the file is a symlink so need to delete the link first just to be safe.
                Image.fromarray(tile).save(save_path)

        return processed_img

    def copy_level(self, level='all', symlink=True, overwrite=False):
        '''
        Copy all tiles from the src to the target
        :param level:       Integer specifying the level, or 'all' meaning all levels
        :param symlink:     Whether to just use symlink to save space.
        :param overwrite:   Whether to overwrite the tiles in the target dir
        '''

        if isinstance(level, str) and level.lower().startswith('all'):
            level = range(self.level_count)
        elif not isinstance(level, list):
            level = [level]

        for lvl in level:
            level_dir_src = os.path.join(self.src_dir, "{}".format(self.level_count - lvl - 1))
            level_dir_dst = os.path.join(self.target_dir, "{}".format(self.level_count - lvl - 1))

            for tilename in os.listdir(level_dir_src):
                tilepath_src = os.path.join(level_dir_src, tilename)
                tilepath_dst = os.path.join(level_dir_dst, tilename)

                if not os.path.exists(level_dir_dst):
                    os.makedirs(level_dir_dst)

                if symlink and overwrite and (not os.path.exists(tilepath_dst)) and (tilepath_src != tilepath_dst):
                    os.symlink(os.path.abspath(tilepath_src), os.path.abspath(tilepath_dst))
                elif (overwrite or (not os.path.exists(tilepath_dst))) and (tilepath_src != tilepath_dst):
                    os.remove(tilepath_dst) if os.path.exists(tilepath_dst) else None
                    shutil.copyfile(tilepath_src, tilepath_dst)

        self.close()

    def downsample_pyramid(self, level_start, level_end=None):
        '''
        Updates all the tiles in target_dir by downsampling the tiles from level_start all the way to level_end.
        '''
        assert (not self.cropped and not self.resized)
        self.copy_level(level=level_start, symlink=False) # Ensures all the tiles from the source had been copied to the target.
        level_end = self.level_count-1 if level_end is None else level_end
        for l in range(level_start, level_end):
            self.update_higher_level(l)
        self.close()

    def update_higher_level(self, l, start_col=0, start_row=0, end_col=None, end_row=None, border=0):
        '''
        Updates the pyramid level l to level l+1
        :param l:
        :param start_col:   reduces the number of tiles that needs to be updated
        :param start_row:
        :param end_col:
        :param end_row:
        :param border: if not None, allows reading regions outside the slide's width/height. Border regions will be greyscale (0-255) given by this parameter.
        :return:
        '''
        assert (not self.cropped and not self.resized)
        assert(l < self.level_count - 1)
        self.copy_level(l+1, overwrite=True)
        try:
            round_y = -1 if self.level_dimensions(l + 1)[1] * np.power(2, l + 1) >= self.height else 0  # Todo: Use level_dimension instead of power to downscale.
            round_x = -1 if self.level_dimensions(l + 1)[0] * np.power(2, l + 1) >= self.width else 0
        except Exception as e:
            round_y = round_x = 0
        # self.delete_layers(l+1)

        max_col, max_row = self.get_max_colrow(l)
        end_col = max_col if end_col is None else max(0, min(max_col, end_col))
        end_row = max_row if end_row is None else max(0, min(max_row, end_row))
        level_dir = os.path.join(self.target_dir, "{}".format(self.level_count - l - 1))

        for i in range(start_col, end_col+1):
            for j in range(start_row, end_row+1):

                try:
                    tile = Image.open(os.path.join(level_dir, "{:d}_{:d}.{:s}".format(i, j, self.format)))
                except:
                    tile = Image.open(os.path.join(os.path.join(self.src_dir, "{}".format(self.level_count - l - 1)), "{:d}_{:d}.{:s}".format(i, j, self.format)))

                if min(tile.size) < 4:
                    break

                _x, _y = self.tile_idx2xy((i, j), level_src=l, level_target=0)

                self.process_region((_x,_y), l+1, (np.int(np.ceil(tile.size[0]/2+round_x)), np.int(np.ceil(tile.size[1]/2+round_y))), self.halfsize(tile), border=border)

    def crop(self, r1, r2, src='target', t=None, v=None, temp_dir='./temp', border=None):
        '''
        Crop the entire image pyramid in target_dir such that the base is bounded by r1, r2
        This must be the last step in the process.
        :param r1: (x1, y1) top left corner
        :param r2: (x2, y2) bottom right corner
        :param src: whether to crop the src or the target
        :param t: New tilesize
        :param v: New overlap
        :param temp_dir: Temporarily stores cropped images in this dir before process finishes
        :param border: if not None, allows reading regions outside the slide's width/height. Border regions will be greyscale (0-255) given by this parameter.
        '''

        assert(not self.cropped and not self.resized)

        t = self.tilesize if t is None else t
        v = self.overlap if v is None else v
        r1, r2 = (np.int(min(r1[0], r2[0])), np.int(min(r1[1], r2[1]))), (np.int(max(r1[0], r2[0])), np.int(max(r1[1], r2[1])))
        if border is None:
            r1 = (np.int(np.maximum(0, np.minimum(r1[0], self.width))), np.int(np.maximum(0, np.minimum(r1[1], self.height))))
            r2 = (np.int(np.maximum(0, np.minimum(r2[0], self.width))), np.int(np.maximum(0, np.minimum(r2[1], self.height))))
        w = r2[0] - r1[0] + 1
        h = r2[1] - r1[1] + 1
        new_level_count = np.int(np.log2(np.maximum(w,h))+2)
        self.target_level_count = new_level_count

        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)


        for l in range(new_level_count):

            if not os.path.exists(os.path.join(temp_dir, "{}".format(new_level_count - l - 1))):
                os.makedirs(os.path.join(temp_dir, "{}".format(new_level_count - l - 1)))

            _r1 = (r1[0] * np.power(0.5, l), r1[1] * np.power(0.5, l))
            _r2 = (r2[0] * np.power(0.5, l), r2[1] * np.power(0.5, l))
            _w = _r2[0] - _r1[0] + 1
            _h = _r2[1] - _r1[1] + 1

            # First, work out how many col/rows there'll be in the new image.
            n_col = np.int(np.ceil(_w / t))
            n_row = np.int(np.ceil(_h / t))

            for i in range(n_col):
                for j in range(n_row):
                    start_r = (np.int(_r1[0]+i*t-v*(i>0)), np.int(_r1[1]+j*t-v*(j>0)))
                    tilesize = [t+v+v*(i>0),t+v+v*(j>0)]
                    if start_r[0] + tilesize[0] > _r2[0]:
                        tilesize[0] = np.int(_r2[0] - start_r[0])
                    if start_r[1] + tilesize[1] > _r2[1]:
                        tilesize[1] = np.int(_r2[1] - start_r[1])

                    if tilesize[0]==0 or tilesize[1]==0:
                        if 'l_min' not in locals():
                            l_min = l
                        break

                    tile = Image.fromarray(self.read_region(start_r, l, tilesize, src=src, mode=2, border=border))

                    # save to temp dir
                    tile.save(os.path.join(temp_dir, "{}".format(new_level_count - l - 1), "{:d}_{:d}.{:s}".format(i, j, self.format)))

        if os.path.exists(self.target_dir):
            shutil.rmtree(self.target_dir)
        shutil.move(temp_dir, self.target_dir)

        if 'l_min' in locals():
            self.blank_layers(range(l_min, new_level_count))
            self.downsample_pyramid(l_min)

        # for i in range(new_level_count - 1):
        #     os.makedirs(os.path.join(self.target_dir, "{}".format(i)))

        # Update the dzi file.
        f = open(self.src, 'r')
        meta = f.read()
        f.close()
        meta = re.sub(r'Overlap="(\d{1,4})', 'Overlap="{}'.format(v), meta)
        meta = re.sub(r'TileSize="(\d{1,4})', 'TileSize="{}'.format(t), meta)
        meta = re.sub(r'Height="(\d{1,6}")', 'Height="{}"'.format(h), meta)
        meta = re.sub(r'Width="(\d{1,6}")', 'Width="{}"'.format(w), meta)
        with open(self.target, 'w') as f:
            f.write(meta)                   # writing to target .dzi file after creating new DZIIO object as the constructor itself copies the dzi file.

        self.cropped = True
        self.close()

    def rotate(self, angle=0, src='target', rot_center=None, tight=True, border=None, temp_dir='./temp'):
        '''
        Rotates the dzi image. if not tight, the image will be the same size as the input; else the image will be cropped so there are no borders.
        :param angle: clockwise, degree (clockwise as the y axis of numpy array is left-handed)
        :param src: whether to rotate the src or the target
        :param rot_center: (x0, y0) center of rotation. Leave blank to rotate about the image's center
        :param tight:   Todo: crops the image after rotation
        :param border: if not None, allows reading regions outside the slide's width/height. Border regions will be greyscale (0-255) given by this parameter.
        :param temp_dir: Temporarily stores images in this dir before process finishes
        :return:
        '''

        assert (not self.cropped and not self.resized)

        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

        os.makedirs(temp_dir)

        for level in range(self.level_count):

            # Centre of rotation
            if rot_center is None:
                x0 = self.level_dimensions(level)[0] / 2.0
                y0 = self.level_dimensions(level)[1] / 2.0
            else:
                x0, y0 = rot_center
            n_col, n_row = self.get_max_colrow(level)
            os.makedirs(os.path.join(temp_dir, "{}".format(self.level_count - level - 1)))

            for i in range(n_col+1):
                for j in range(n_row+1):
                    start_r = (i * self.tilesize - self.overlap * (i > 0), j * self.tilesize - self.overlap * (j > 0))
                    tilesize = [self.tilesize + self.overlap + self.overlap * (i > 0), self.tilesize + self.overlap + self.overlap * (j > 0)]
                    if start_r[0] + tilesize[0] > self.level_dimensions(level)[0]:
                        tilesize[0] = self.level_dimensions(level)[0] - start_r[0]
                    if start_r[1] + tilesize[1] > self.level_dimensions(level)[1]:
                        tilesize[1] = self.level_dimensions(level)[1] - start_r[1]

                    # Read a larger region, rotate it, and crop out the area with image.
                    rotmat = cv2.getRotationMatrix2D((0, 0), angle, 1.0)
                    start_r_read0 = np.matmul(rotmat[:, :2], (start_r[0] - x0, start_r[1] - y0)) + np.array([x0, y0])
                    start_r_read1 = np.matmul(rotmat[:, :2], (start_r[0] + tilesize[0] - x0, start_r[1] - y0)) + np.array([x0, y0])
                    start_r_read2 = np.matmul(rotmat[:, :2], (start_r[0] - x0, start_r[1] + tilesize[1] - y0)) + np.array([x0, y0])
                    start_r_read3 = np.matmul(rotmat[:, :2], (start_r[0] + tilesize[0] - x0, start_r[1] + tilesize[1] - y0)) + np.array([x0, y0])
                    start_r_read = (np.min([start_r_read0, start_r_read1, start_r_read2, start_r_read3], axis=0)).astype(np.int)
                    tilesize_read = np.ceil(np.max([start_r_read0, start_r_read1, start_r_read2, start_r_read3], axis=0) - start_r_read).astype(np.int)

                    tile = self.read_region(self.map_xy(start_r_read, level), level, tilesize_read, src=src, border=border)
                    rotated = utils.rotate_bound(tile, angle)
                    read_centre = (rotated.shape[1]/2, rotated.shape[0]/2)
                    rotated = rotated[np.int(read_centre[1] - tilesize[1] / 2):np.int(read_centre[1] + tilesize[1] / 2),
                              np.int(read_centre[0] - tilesize[0] / 2):np.int(read_centre[0] + tilesize[0] / 2)]

                    rotated = Image.fromarray(rotated)
                    rotated.save(os.path.join(temp_dir, "{}".format(self.level_count - level - 1), "{:d}_{:d}.{:s}".format(i, j, self.format)))

        if os.path.exists(os.path.join(self.target_dir)):
            shutil.rmtree(os.path.join(self.target_dir))
        shutil.move(temp_dir, self.target_dir)

        self.close()

    def rotcrop(self, center, wh, angle, border=None, temp_dir1='./temp', temp_dir2='./temp2', temp_dir3='./temp3'):
        """
        Rotate and crop image to a given rectangle.
        # Todo Code may not be optimal as it is based on chaining existing functions self.crop() and self.rotate().
        # 1) Crop image first so that the width is >= width of new box
        # 2) Rotate image about the centre of the minAreaRect
        # 3) Crop image again.

        :param center: center of the rectangle
        :param wh: (w, h) width and height of the rectangle
        :param angle: Angle in which the rectangle is rotated by (deg), clockwise is positive
        :param border:
        :return: if not None, allows reading regions outside the slide's width/height. Border regions will be greyscale (0-255) given by this parameter.
        """

        phi = np.arctan(wh[1]/wh[0])
        d = np.sqrt(wh[1]*wh[1] + wh[0]*wh[0])/2  # Diagonal of rect
        proj_w = d*np.maximum(np.abs(np.cos(angle*np.pi/180+phi)), np.abs(np.cos(angle*np.pi/180-phi))) # Projected half-width
        proj_h = d*np.maximum(np.abs(np.sin(angle*np.pi/180+phi)), np.abs(np.sin(angle*np.pi/180-phi))) # Projected half-height
        w1 = np.maximum(wh[0] / 2, proj_w)
        h1 = np.maximum(wh[1] / 2, proj_h)

        if Path(temp_dir1).exists():  shutil.rmtree(temp_dir1)
        if Path(temp_dir2).exists():  shutil.rmtree(temp_dir2)
        if Path(temp_dir3).exists():  shutil.rmtree(temp_dir3)

        self.crop((center[0]-w1,center[1]-h1), (center[0]+w1,center[1]+h1), temp_dir=temp_dir1, border=border)
        shutil.move(self.target_dir, Path(temp_dir2)/Path(self.target_dir).name)
        shutil.copy(self.target, Path(temp_dir2) / self.target.name)

        cropped_dzi = DZIIO(Path(temp_dir2) / self.target.name, target=Path(temp_dir3)/self.target.name)
        cropped_dzi.rotate(angle, border=border, temp_dir=temp_dir1)

        cropped_rotated_dzi = DZIIO(Path(temp_dir3)/self.target.name, target=self.target)
        cropped_rotated_dzi.crop((w1-wh[0]/2, h1-wh[1]/2), (w1+wh[0]/2, h1+wh[1]/2), temp_dir=temp_dir1, border=border)

        del cropped_dzi
        del cropped_rotated_dzi

        shutil.rmtree(temp_dir2)
        shutil.rmtree(temp_dir3)

        self.cropped = True

    def resize(self, new_size, mpp=None, src='target', temp_dir='./temp'):
        '''
        Todo:
        Resize the image pyramid so that the base level fits within new_size == (new_width, new_height)
        Alternatively, if the mpp is given and it is found in the metadata, the pyramid will be scaled to the given mpp

        :param src: whether to rotate the src or the target
        :param temp_dir: Temporarily stores images in this dir before process finishes
        :return:
        '''

        assert (not self.cropped and not self.resized)

        self.resized = True
        print("Unfinished")

    def blank_layers(self, levels, random=True):
        '''
        Lays blank tiles at a level
        :param levels: list with levels >0
        '''

        for level in levels:
            w = np.maximum(np.int(self.level_dimensions(0)[0] * np.power(2.0, -level)), 1)
            h = np.maximum(np.int(self.level_dimensions(0)[1] * np.power(2.0, -level)), 1)
            n_col = np.int(np.ceil(w / self.tilesize))
            n_row = np.int(np.ceil(h / self.tilesize))

            try:
                level_dir = os.path.join(self.target_dir, "{}".format(self.target_level_count-level-1)) # In some cases the target may have different no. of levels from the src.
            except:
                level_dir = os.path.join(self.target_dir, "{}".format(self.level_count - level - 1))  # Note that dzi orders dir's magnification in ascending order

            for i in range(n_col):
                for j in range(n_row):
                    start_r = (i * self.tilesize - self.overlap * (i > 0), j * self.tilesize - self.overlap * (j > 0))
                    tilesize = [self.tilesize + self.overlap + self.overlap * (i > 0), self.tilesize + self.overlap + self.overlap * (j > 0)]
                    if start_r[0] + tilesize[0] > w:
                        tilesize[0] = w - start_r[0]
                    if start_r[1] + tilesize[1] > h:
                        tilesize[1] = h - start_r[1]
                    tile = (np.ones((tilesize[1], tilesize[0], 3)) * 255 * np.random.random() * random).astype(np.uint8)
                    # cv2.putText(tile, '{},{},{}'.format(level,i,j), (int(tilesize[0]/2), tilesize[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0))
                    Image.fromarray(tile).save(os.path.join(level_dir, "{:d}_{:d}.{:s}".format(i, j, self.format)))

    def clean_target(self, levels=None, supress_warning=False):
        '''
        Deletes all files in the target directory. Use with caution!
        :param levels: a list of the levels to be deleted. Delete everything if None
        :param supress warning:
        '''

        if not supress_warning:
            ans = input("Do you really want to delete all files in the target directory? [y/N]")

            if ans.lower() != 'y':
                print("Abort")
                return None

        if levels is None and os.path.exists(self.target_dir):
            shutil.rmtree(self.target_dir)
        elif os.path.exists(self.target_dir):
            for level in levels:
                if os.path.exists(os.path.join(self.target_dir, "{}".format(self.level_count - level - 1))):
                    shutil.rmtree(os.path.join(self.target_dir, "{}".format(self.level_count - level - 1)))

    # --------------------- Deprecated Functions --------------------------
    @utils.deprecated('update_pyramid() has been renamed to downsample_pyramid()')
    def update_pyramid(self, level_start, level_end=None):
        '''
        This has been renamed to downsample_pyramid.
        '''
        self.downsample_pyramid(level_start, level_end=level_end)


class DZISequential(object):
    '''
    A sequential container that allows complex operations to multiple dzi files.
    The first input must be dzi. Other inputs can be dzi, numpy arrays, or constant.
    Input with mpp information will be scaled to the first input according to its mpp.
    Otherwise it will be scaled to best-fit the first input.

    Example:
    >>> inputs = (dzi1, dzi2, dzi3)
    >>> fn = lambda (x, y, z): y*x + (1-y)*z  # x could be a segmentation mask; y could be a mask of alpha values; z could be the original image.
    >>> seq = DZISequential(inputs, fn)
    >>> seq.evaluate()
    '''
    def __init__(self, inputs, fn):
        self.define_inputs(inputs)
        self.define_operations(fn)

    def define_operations(self, fn):
        self.fn = fn

    def define_inputs(self, inputs):
        self.inputs = inputs

    def evaluate(self, temp_dir='./temp'):

        # First check if both images contain mpp information.
        # If both contains mpp info, scale them to the same mpp, aligned at (0,0)
        # Else scale all the other inputs to the scale  of the first.

        has_mpp = True if all([input.mpp is not None for input in self.inputs]) else False

        if has_mpp:
            mpps = [np.float(input.mpp) for input in self.inputs]
            scalefactors = [mpps[0]/mpp for mpp in mpps]
        else:
            scalefactors = [1.0]
            for input in self.inputs[1:]:
                scalefactors.append(np.maximum(input.level_dimensions(0)[0]/self.inputs[0].level_dimensions(0)[0],
                                               input.level_dimensions(0)[1]/self.inputs[0].level_dimensions(0)[1]))

        max_col, max_row = self.inputs[0].get_max_colrow(0)
        if not os.path.exists(os.path.join(self.inputs[0].target_dir, "{}".format(self.inputs[0].level_count - 1))):
            os.makedirs(os.path.join(self.inputs[0].target_dir, "{}".format(self.inputs[0].level_count - 1)))

        for i in range(max_col + 1):
            for j in range(max_row + 1):

                loc_main = np.array(self.inputs[0].tile_idx2xy((i,j)))
                input_img = Image.open(os.path.join(self.inputs[0].src_dir, "{}/{}_{}.{}".format(self.inputs[0].level_count - 1, i, j, self.inputs[0].format)))
                input_imgs = [np.array(input_img).astype(np.float)]

                for input, scale in zip(self.inputs[1:], scalefactors[1:]):

                    location = (loc_main * scale).astype(np.int)
                    size = np.array([input_imgs[0].shape[1] * scale, input_imgs[0].shape[0] * scale]).astype(np.int)
                    _input = Image.fromarray(input.read_region(location, 0, size, border=0)).resize(input_img.size)
                    input_imgs.append(np.array(_input).astype(np.float))

                output = self.fn(*input_imgs).astype(np.uint8)
                Image.fromarray(output).save(os.path.join(self.inputs[0].target_dir, "{}/{}_{}.{}".format(self.inputs[0].level_count - 1, i, j, self.inputs[0].format)))

        self.inputs[0].downsample_pyramid(0)
        self.inputs[0].close()


def DZI_IO(src, target=None, clean_target=False):
    warnings.warn("DZI_IO class has been renamed to DZIIO", DeprecationWarning)
    return DZIIO(src, target=target, clean_target=clean_target)


def DZI_Sequential(inputs, fn):
    warnings.warn("DZI_Sequential class has been renamed to DZISequential", DeprecationWarning)
    return DZISequential(inputs, fn)

