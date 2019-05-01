# DZI_IO

Class for reading/writing directly into Deep Zoom (.dzi) file hierarchy.

1. Convert .ndpi / .svs files into .dzi using *deepzoom_tile.py*
    - This differs from the [original](https://github.com/openslide/openslide-python/blob/master/examples/deepzoom/deepzoom_tile.py) as the current *deepzoom_tile.py* saves the slide properties as a json file inside the dzi_files directory.
    - Example usage (see parser): deepzoom_tile.py test.ndpi -o outputfilename -e overlap -s 254 -e 1

2. Process the .dzi files
    - See *demo/dzi_io_demo.py* and *demo/cnn_demo.py* for example usage.