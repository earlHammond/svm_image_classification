
import os
import sys
import random
import copy
import numpy as np
import pandas as pd

import ImageProcessing
import Preprocessing
import Clustering
import Models
import FileUtils
import Kernels

from sklearn.cross_validation import KFold

NP_SAVE_LOCATION = "data"
CLUSTER_CENTER_NAME = "cluster-centers"
LABELED_PREDICTORS = "labeled-predictors"
LABELED_RESPONSE = "labeled-response"
UNLABELED_PREDICTORS = "unlabeled-predictors"
UNLABELED_RESPONSE = "unlabeled-response"


def main(run_name="test",
         k=50,
         surf_function='partial',
         sort_data=True,
         run_image_crop=True,
         extract_features=True,
         create_labeled_images=True,
         train_model=True,
         predict_model=True):

    if surf_function == 'dense':
        surf = ImageProcessing.run_dense_surf
    elif surf_function == 'partial':
        surf = ImageProcessing.run_surf
    else:
        sys.exit(1)

    all_image_path = sys.argv[1]
    training_csv_path = sys.argv[2]
    output_path = sys.argv[3]

    all_images_cropped_path = os.path.join(output_path, "CroppedImages/All")
    training_images_cropped_path = os.path.join(output_path, "CroppedImages/Training")
    test_images_cropped_path = os.path.join(output_path, "CroppedImages/Test")

    if run_image_crop:
        print "Cropping images"
        all_images = get_images(all_image_path)
        all_images_cropped_path = FileUtils.make_dir(all_images_cropped_path, overwrite=True)
        # all_images_cropped = crop_images(all_images, all_images_cropped_path, k)
        all_images_cropped = crop_images_hist(all_images, all_images_cropped_path)
    else:
        all_images_cropped = get_images(all_images_cropped_path)

    if sort_data:
        print "Sorting images"
        training_images_cropped_path = FileUtils.make_dir(training_images_cropped_path)
        test_images_cropped_path = FileUtils.make_dir(test_images_cropped_path)
        sort_training_files(training_csv_path, all_images_cropped_path,
                            training_images_cropped_path, test_images_cropped_path)

    if extract_features:
        print "Extracting features"
        if create_labeled_images:
            images = get_images(training_images_cropped_path)
        else:
            images = all_images_cropped

        feature_stream = extract_image_features_stream(surf, images)
        cluster_centers = Clustering.cluster_key_points_from_stream(feature_stream, k)
        save_np_file(cluster_centers, CLUSTER_CENTER_NAME, run_name)

    if create_labeled_images:
        print "Build histograms for all images"
        cluster_centers = load_np_file(CLUSTER_CENTER_NAME, run_name)
        labeled_images = get_images(training_images_cropped_path)
        labeled_X, labeled_y = Clustering.build_histograms_from_features(surf, k, cluster_centers, labeled_images, all_images_cropped_path)
        save_np_file(labeled_X, LABELED_PREDICTORS, run_name)
        save_np_file(labeled_y, LABELED_RESPONSE, run_name)

    if train_model:
        labeled_X = load_np_file(LABELED_PREDICTORS, run_name)
        labeled_y = load_np_file(LABELED_RESPONSE, run_name)
        kernels = ['poly', 'rbf', 'sigmoid', Kernels.laplacian_kernel]
        penalties = [0.1, 1, 100, 1000, 10000]

        scores = {}
        # kfold = Models.custom_k_folds(labeled_y, n_folds=3, min_response=1)
        kfold = KFold(len(labeled_y), n_folds=5)
        print "Training SVM Model"
        clf_score = Models.train_forest(labeled_X, labeled_y, k, kfold)
        print clf_score
        print "Training SVM Model"
        for kernel in kernels:
            for penalty in penalties:
                score = Models.train_svm(labeled_X, labeled_y, kernel, penalty, kfold)
                scores[(kernel, penalty)] = score

        for run in scores:
            print run, scores[run]

        data_frame = pd.DataFrame(scores.items())
        data_frame.to_pickle("data/output_%s" % run_name)

    if predict_model:
        unlabled_X = load_np_file(UNLABELED_PREDICTORS, run_name)
        unlabled_y = load_np_file(UNLABELED_RESPONSE, run_name)


def save_np_file(data, file_name, run_name):
    file_path = os.path.join(NP_SAVE_LOCATION, "%s_%s" % (file_name, run_name))

    if not os.path.exists(NP_SAVE_LOCATION):
        os.makedirs(NP_SAVE_LOCATION)

    with open(file_path, 'wb') as out_file:
        np.save(out_file, data)


def load_np_file(file_name, run_name):
    file_path = os.path.join(NP_SAVE_LOCATION, "%s_%s" % (file_name, run_name))

    with open(file_path, 'rb') as in_file:
        return np.load(in_file)


def get_images(image_folder):
    images = []

    for root, folders, files in os.walk(image_folder):
        for f in files:
            if os.path.splitext(f)[1] == '.jpg':
                images.append(os.path.join(root, f))
    return images


def sort_training_files(training_csv_path, all_image_path, training_image_path, test_image_path):
    training_lookup = Preprocessing.read_training_file(training_csv_path)
    Preprocessing.sort_training_images(training_lookup, all_image_path, training_image_path, test_image_path)


def crop_images_hist(images, cropped_folder):
    cropped_images = []

    for image in images:
        output_path = os.path.join(cropped_folder, os.path.split(image)[1])
        ImageProcessing.crop_by_color_histogram(image, output_path)
        cropped_images.append(output_path)

    return cropped_images


def crop_images(surf_function, images, cropped_folder, k):
    cropped_images = []

    for image in images:
        key_points = surf_function(image)[0]
        output_path = os.path.join(cropped_folder, os.path.split(image)[1])
        ImageProcessing.crop_image(image, output_path, key_points, k)
        cropped_images.append(output_path)

    return cropped_images


def sample_images(images, sample_percentage):
    images = copy.deepcopy(images)
    random.shuffle(images)
    count = int(len(images) * sample_percentage)
    return images[-count:]


def extract_image_features(surf_function, image_list):
    key_points = {}
    features = {}
    for image in image_list:
        print "Extracting SURF for image %s" % image
        kp, des = surf_function(image)

        if kp is not None and des is not None:
            key_points[image] = kp
            features[image] = np.asarray(des)

    return key_points, features


def extract_image_features_stream(surf_function, image_list):
    for image in image_list:
        print "Extracting SURF for image %s" % image
        kp, des = surf_function(image)

        if kp is not None and des is not None:
            yield kp, np.asarray(des)

if __name__ == "__main__":
    main(run_name="cal_k100_dense",
         k=100,
         surf_function='dense',
         sort_data=False,
         run_image_crop=False,
         extract_features=False,
         create_labeled_images=False,
         train_model=True,
         predict_model=False)
