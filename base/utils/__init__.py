import importlib
from .base_visualizer import BaseVisualizer


def find_visualizer_using_name(task_name):
    # Given the option --visualizer_name [visualizername],
    # the file "data/visualizername_visualizer.py"
    # will be imported.
    try:
        task_module = importlib.import_module(task_name)
        visualizer_filename = task_name + ".utils.visualizer"
        visualizerlib = importlib.import_module(visualizer_filename, package=task_module)
    except ModuleNotFoundError as err1:
            try:
                # if module not found, attempt to load from base
                task_name = 'base'
                task_module = importlib.import_module(task_name)
                visualizer_filename = task_name + ".utils.visualizer"
                visualizerlib = importlib.import_module(visualizer_filename, package=task_module)
            except ModuleNotFoundError:
                if not err1.args:
                    err1.args = ('',)
                err1.args = err1.args + (f"{task_name}.utils contains no file 'visualizer.py'",)
                raise err1
    except ImportError as importerr:
        if not importerr.args:
            importerr.args = ('',)
        importerr.args = importerr.args + (f"Module {task_name} not found.",)
        raise

    def is_subclass(subclass, superclass):
        return next(iter(subclass.__bases__)).__module__.endswith(superclass.__module__)

    visualizer = None
    target_visualizer_name = task_name + 'visualizer'
    for name, cls in visualizerlib.__dict__.items():
        if name.lower() == target_visualizer_name.lower():
            if is_subclass(cls, BaseVisualizer) or any(is_subclass(cls_b, BaseVisualizer) for cls_b in cls.__bases__):
                visualizer = cls

    if visualizer is None:
        raise NotImplementedError("In visualizer.py, there should be a subclass of BaseVisualizer with class name that matches {} in lowercase.".format(
            target_visualizer_name))

    return visualizer


def create_visualizer(opt):
    dataset = find_visualizer_using_name(opt.task)
    instance = dataset(opt)
    return instance
