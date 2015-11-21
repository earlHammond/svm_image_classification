
import os
import sys
import numpy as np

import ImageProcessing
import Preprocessing
import Clustering
import Models
import FileUtils
import Kernels

NP_SAVE_LOCATION = "data"
CLUSTER_CENTER_NAME = "cluster-centers"
LABELED_PREDICTORS = "labeled-predictors"
LABELED_RESPONSE = "labeled-response"
UNLABELED_PREDICTORS = "unlabeled-predictors"
UNLABELED_RESPONSE = "unlabeled-response"


def main(run_name="test",
         k=50,
         sort_data=True,
         run_image_crop=True,
         extract_all_features=True,
         create_labeled_images=True,
         create_unlabeled_images=True,
         train_model=True,
         predict_model=True):

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
        all_images_cropped = crop_images(all_images, all_images_cropped_path, k)
    else:
        all_images_cropped = get_images(all_images_cropped_path)

    if sort_data:
        print "Sorting images"
        training_images_cropped_path = FileUtils.make_dir(training_images_cropped_path)
        test_images_cropped_path = FileUtils.make_dir(test_images_cropped_path)
        sort_training_files(training_csv_path, all_images_cropped_path,
                            training_images_cropped_path, test_images_cropped_path)

    if extract_all_features:
        if create_labeled_images and not create_unlabeled_images:
            images = get_images(training_images_cropped_path)
        else:
            images = all_images_cropped

        print "Extracting features"
        all_features = extract_image_features(images)[1]
        cluster_centers = Clustering.cluster_key_points(all_features, k)
        save_np_file(cluster_centers, CLUSTER_CENTER_NAME, run_name)

    if create_labeled_images:
        cluster_centers = load_np_file(CLUSTER_CENTER_NAME, run_name)
        labeled_images = get_images(training_images_cropped_path)
        labeled_features = extract_image_features(labeled_images)[1]
        labeled_X, labeled_y = Clustering.build_histograms_from_features(k, cluster_centers, labeled_features, all_images_cropped_path)
        save_np_file(labeled_X, LABELED_PREDICTORS, run_name)
        save_np_file(labeled_y, LABELED_RESPONSE, run_name)

    if create_unlabeled_images:
        cluster_centers = load_np_file(CLUSTER_CENTER_NAME, run_name)
        unlabeled_images = get_images(test_images_cropped_path)
        unlabeled_features = extract_image_features(unlabeled_images)[1]
        unlabled_X, unlabled_y = Clustering.build_histograms_from_features(k, cluster_centers, unlabeled_features)
        save_np_file(unlabled_X, UNLABELED_PREDICTORS, run_name)
        save_np_file(unlabled_y, UNLABELED_RESPONSE, run_name)

    if train_model:
        labeled_X = load_np_file(LABELED_PREDICTORS, run_name)
        labeled_y = load_np_file(LABELED_RESPONSE, run_name)
        kernels = ['poly', 'rbf', 'sigmoid', Kernels.laplacian_kernel]
        penalties = [0.1, 1, 100, 1000, 10000]

        scores = {}
        for kernel in kernels:
            for penalty in penalties:
                score = Models.train_svm(labeled_X, labeled_y, kernel, penalty)
                scores[(kernel, penalty)] = score

        for run in scores:
            print run, scores[run]

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


def crop_images(images, cropped_folder, k):
    cropped_images = []

    for image in images:
        key_points = ImageProcessing.run_surf(image)[0]
        output_path = os.path.join(cropped_folder, os.path.split(image)[1])
        ImageProcessing.crop_image(image, output_path, key_points, k)
        cropped_images.append(output_path)

    return cropped_images


def extract_image_features(image_list):
    key_points = {}
    features = {}
    for image in image_list:
        print "Extracting SURF for image %s" % image
        kp, des = ImageProcessing.run_surf(image)

        if kp is not None and des is not None:
            key_points[image] = kp
            features[image] = np.asarray(des)

    return key_points, features


if __name__ == "__main__":
    main(run_name="test",
         k=50,
         sort_data=True,
         run_image_crop=True,
         extract_all_features=True,
         create_labeled_images=True,
         create_unlabeled_images=False,
         train_model=True,
         predict_model=False)
