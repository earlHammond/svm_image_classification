import os
import sys

import numpy as np
import pandas as pd

import ImageProcessing
import Preprocessing
import Clustering
import Models
import Histogram
import FileUtils
import CrossValidation
import Kernels
import Vectorization

from collections import OrderedDict

NP_SAVE_LOCATION = "data"
CLUSTER_CENTER_NAME = "cluster-centers"
LABELED_PREDICTORS = "labeled-predictors"
LABELED_RESPONSE = "labeled-response"
UNLABELED_PREDICTORS = "unlabeled-predictors"
UNLABELED_RESPONSE = "unlabeled-response"


def main(run_name="test",
         k=50,
         surf_spread='dense',
         stratified_kfold=False,
         sort_data=True,
         run_image_crop=True,
         extract_features=True,
         create_labeled_images=True,
         build_models=True):

    all_image_path = sys.argv[1]
    training_csv_path = sys.argv[2]
    output_path = sys.argv[3]

    cropped_images_path = os.path.join(output_path, "CroppedImages")
    sorted_images_path = os.path.join(output_path, "SortedImages")

    if run_image_crop:
        print "Cropping images"
        all_images = FileUtils.get_images(all_image_path)
        all_images_cropped_path = FileUtils.make_dir(cropped_images_path, overwrite=True)
        crop_images(all_images, all_images_cropped_path)

    if sort_data:
        print "Sorting images"
        sorted_folder = FileUtils.make_dir(sorted_images_path, overwrite=True)
        sort_training_files(training_csv_path, cropped_images_path, sorted_folder)

    if extract_features:
        print "Building codebook"
        images = FileUtils.get_images(sorted_images_path)

        feature_stream = extract_image_features_stream(surf_spread, images)
        cluster_centers = Clustering.cluster_key_points_from_stream(feature_stream, k)
        save_np_file(cluster_centers, CLUSTER_CENTER_NAME, run_name)

    if create_labeled_images:
        print "Build histograms for all images"
        cluster_centers = load_np_file(CLUSTER_CENTER_NAME, run_name)
        labeled_images = FileUtils.get_images(sorted_images_path)
        labeled_X, labeled_y = Histogram.build_image_histograms(surf_spread, k, cluster_centers, labeled_images)
        save_np_file(labeled_X, LABELED_PREDICTORS, run_name)
        save_np_file(labeled_y, LABELED_RESPONSE, run_name)

    if build_models:
        labeled_X = load_np_file(LABELED_PREDICTORS, run_name)
        labeled_y = load_np_file(LABELED_RESPONSE, run_name)

        if stratified_kfold:
            kfold = CrossValidation.get_stratified_fold(labeled_y)
        else:
            kfold = CrossValidation.get_kfold(labeled_y)

        scores = OrderedDict()
        forest_score = Models.train_forest(labeled_X, labeled_y, k, kfold, Vectorization.merge_and_vectorize_features)
        scores["Random Forest"] = forest_score

        boost_score = Models.train_gradient_boosting_classifier(labeled_X, labeled_y, k, kfold, Vectorization.merge_and_vectorize_features)
        scores["Gradient Boosting"] = boost_score

        print "Training SVM Model"
        kernels = ['poly', 'rbf', 'sigmoid', Kernels.laplacian_kernel]
        penalties = [0.1, 1, 100, 1000, 10000]
        for kernel in kernels:
            for penalty in penalties:
                score = Models.train_svm(labeled_X, labeled_y, kernel, penalty, kfold, Vectorization.merge_and_vectorize_features)
                scores["%s_%s_%i" % ('SVM', str(kernel), penalty)] = score

        for run in scores:
            print run, scores[run], np.mean(scores[run])

        data_frame = pd.DataFrame(scores.items())
        data_frame.to_pickle("data/output_%s" % run_name)


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


def sort_training_files(training_csv_path, all_image_path, sorted_folder):
    training_lookup = Preprocessing.read_training_file(training_csv_path)
    Preprocessing.sort_training_images(training_lookup, all_image_path, sorted_folder)


def crop_images(images, cropped_folder):
    for image in images:
        output_path = os.path.join(cropped_folder, os.path.split(image)[1])
        ImageProcessing.crop_by_color_histogram(image, output_path)


def extract_image_features_stream(spread, image_list):
    for image in image_list:
        print "Extracting SURF for image %s" % image
        kp, des = ImageProcessing.run_surf(image, spread=spread)

        if kp is not None and des is not None:
            yield kp, np.asarray(des)

if __name__ == "__main__":
    main(run_name="test",
         k=100,
         surf_spread='dense',
         stratified_kfold=True,
         sort_data=False,
         run_image_crop=False,
         extract_features=False,
         create_labeled_images=False,
         build_models=True)
