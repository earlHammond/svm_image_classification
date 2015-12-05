import os
import csv
import shutil

import FileUtils


def read_training_file(path):
    training_lookup = {}
    with open(path, 'rb') as training_file:
        for line in csv.reader(training_file):
            training_lookup[line[0]] = line[1]
    return training_lookup


def sort_training_images(training_lookup, all_image_path, training_path, test_path):
    for file_name in os.listdir(all_image_path):
        source_path = os.path.join(all_image_path, file_name)

        if file_name in training_lookup:
            target_folder = FileUtils.make_dir(os.path.join(training_path, training_lookup[file_name]))
            target_path = os.path.join(target_folder, file_name)
        else:
            target_path = os.path.join(test_path, file_name)

        shutil.copy(source_path, target_path)

