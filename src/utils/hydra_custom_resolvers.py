from omegaconf import OmegaConf
import math
import os


def get_dict_except(cfg, *keys):
    breakpoint()
    return {k: v for k, v in cfg.items() if k not in keys}

def add_args(*args):
    return sum(float(x) for x in args)


def add_args_int(*args):
    return int(sum(float(x) for x in args))


def multiply_args(*args):
    return math.prod(float(x) for x in args)


def multiply_args_int(*args):
    return int(math.prod(float(x) for x in args))


def floor_division(dividend, divisor):
    return dividend // divisor

def float_division(dividend, divisor):
    return dividend / divisor

def num_files_in_directory(path):
    isExist = os.path.exists(path)
    if not isExist:
        return 0

    isDir = os.path.isdir(path)
    if not isDir:
        raise Exception(f"Path `{path}` does not correspond to a directory!")

    ls_dir = os.listdir(path)
    return len(ls_dir)


def path_to_python_executable():
    import sys

    return sys.executable


def best_ckpt_path_retrieve(path):
    isExist = os.path.exists(path)
    if not isExist:
        raise Exception(f"Path `{path}` does not exist!")

    isDir = os.path.isdir(path)
    if not isDir:
        raise Exception(f"Path `{path}` does not correspond to a directory!")

    best_ckpt_file_path = os.path.join(path, "best_ckpt_path.txt")
    best_ckpt_exists = os.path.isfile(best_ckpt_file_path)
    if not best_ckpt_exists:
        raise Exception(f"File `{best_ckpt_file_path}` does not exist. trainer.fit() might have crashed before saving the path to the best ckpt!")

    with open(best_ckpt_file_path, "r") as f:
        best_ckpt_name = f.readlines()[0]

    return os.path.join(path, "checkpoints", best_ckpt_name)


OmegaConf.register_new_resolver("add", add_args)
OmegaConf.register_new_resolver("mult", multiply_args)

OmegaConf.register_new_resolver("add_int", add_args_int)
OmegaConf.register_new_resolver("mult_int", multiply_args_int)

OmegaConf.register_new_resolver("floor_div", floor_division)
OmegaConf.register_new_resolver("float_div", float_division)

OmegaConf.register_new_resolver("num_files", num_files_in_directory)

OmegaConf.register_new_resolver("path_to_python_executable", path_to_python_executable)

# OmegaConf.register_new_resolver("get_dict_except", get_dict_except)