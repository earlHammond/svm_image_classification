
import os
import shutil


def make_dir(dir_path, overwrite=False):
    if overwrite and os.path.exists(dir_path):
        shutil.rmtree(dir_path)

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    return dir_path
