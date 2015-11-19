
import os
import sys
import numpy as np

import ImageProcessing
import Preprocessing
import Clustering
import Models
import FileUtils


def main(k, sort_data, run_image_crop, extract_all_features, train_model, classify_images):
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
        print "Extracting features"
        all_features = extract_image_features(all_images_cropped)[1]
        cluster_centers = Clustering.cluster_key_points(all_features, k)
        np.save("cluster_centers", cluster_centers)
    else:
        cluster_centers = np.load("cluster_centers")

    if train_model:
        labeled_images = get_images(training_images_cropped_path)
        labeled_X, labeled_y = Clustering.build_histograms_from_features(k, cluster_centers, labeled_images)
        np.save("labeled_histograms", labeled_X)
        np.save("labeled_response", labeled_y)
    else:
        labeled_X = np.load("labeled_histograms")
        labled_y = np.load("labeled_response")

    if classify_images:
        unlabeled_images = get_images(test_images_cropped_path)
        unlabled_X, unlabled_y = Clustering.build_histograms_from_features(k, cluster_centers, unlabeled_images)
        np.save("unlabeled_histograms", unlabled_X)
        np.save("unlabeled_response", unlabled_y)
    else:
        unlabled_X = np.load("test_histograms")
        unlabled_y = np.load("test_response")




    svm = Models.train_svm(training_X, training_y, 'rbf', 1.0, 0.0)


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
        print "Extracting SIFT for image %s" % image
        kp, des = ImageProcessing.run_surf(image)
        key_points[image] = kp
        features[image] = np.asarray(des)

    return key_points, features


if __name__ == "__main__":
    main(50, False, False)
