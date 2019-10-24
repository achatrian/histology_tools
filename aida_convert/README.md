# AIDA convert tools
Contains annotation builder class to convert sequences of x,y points into AIDA-readable annotations.
 
Basic usage:
```python
from histology_tools.aida_conver.annotation_builder import AnnotationBuilder

paths = [...]  # list of paths, i.e. sequences of x,y points
my_classes = ['class0']
annotation = AnnotationBuilder('slide_1', 'my_project', my_classes)
for path in paths:
    annotation.add_item('class0', 'path')
    annotation.add_segments_to_last_item(path)
annotation.dump_to_json('~/my/data/dir')
```
