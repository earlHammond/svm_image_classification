import os
import csv
import shutil

from src import FileUtils


def read_training_file(path):
    training_lookup = {}
    if os.path.exists(path):
        with open(path, 'rb') as training_file:
            for line in csv.reader(training_file):
                training_lookup[line[0]] = line[1]
    return training_lookup


def sort_training_images(training_lookup, image_path, output_path):
    image_files = FileUtils.get_images(image_path)
    for source_path in image_files:
            if training_lookup:
                image_id = training_lookup[os.path.split(source_path)[1]]
            else:
                image_id = os.path.split(os.path.dirname(source_path))[1]

            target_folder = FileUtils.make_dir(os.path.join(output_path, image_id))
            target_path = os.path.join(target_folder, os.path.split(source_path)[1])

            shutil.copy(source_path, target_path)

