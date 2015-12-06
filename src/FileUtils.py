
import os
import shutil


def make_dir(dir_path, overwrite=False):
    if overwrite and os.path.exists(dir_path):
        shutil.rmtree(dir_path)

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    return dir_path


def get_images(image_folder):
    images = []

    for root, folders, files in os.walk(image_folder):
        for f in files:
            if os.path.splitext(f)[1] == '.jpg':
                images.append(os.path.join(root, f))
    return images
