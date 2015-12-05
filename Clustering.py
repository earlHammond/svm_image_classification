import os
import math
import numpy as np

import ImageProcessing

from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler


def cluster_key_points_from_stream(image_data, k):
    print "Clustering Image Features to %s clusters" % k
    km = MiniBatchKMeans(init='k-means++', n_clusters=k, max_iter=50, batch_size=20)
    descriptors = []
    for index, data in enumerate(image_data):
        descriptors.append(data[1])

        if index % 100:
            descriptors = np.vstack(descriptors)
            km.partial_fit(descriptors)
            descriptors = []

    return km.cluster_centers_


def cluster_key_points(descriptors, k):
    print "Clustering Image Features to %s clusters" % k
    km = KMeans(init='k-means++', n_clusters=k, max_iter=100)
    descriptors = np.vstack(descriptors.values())
    km.fit(descriptors)

    return km.cluster_centers_


def build_histograms_from_features(surf_function, k, cluster_centers, descriptors_lookup, original_path=None):
    km = KMeans(n_clusters=k, max_iter=100, init=cluster_centers, n_init=1)

    X = []
    y = []
    for image_path in descriptors_lookup:
        print "Building Histogram for %s" % image_path
        image_id = os.path.split(os.path.dirname(image_path))[1]

        img = cv2.imread(img_path, cv2.CV_LOAD_IMAGE_GRAYSCALE)
        sections = ImageProcessing.divide_image(image_path)
        kp, des = surf_function(image_path)

        
        surf_value = 400
        attempts = 0
        completed = False

        while attempts < 20 and not completed:
            try:
                descriptors = np.vstack(des)
                km.fit(descriptors)
                completed = True
            except:
                print "Re-running %s with SURF value set to: %i" % (image_path, surf_value)
                surf_value *= 0.80
                attempts += 1
                des = surf_function(image_path, surf_value)[1]

            if attempts == 10 and original_path:
                new_image_path = os.path.join(original_path, os.path.split(image_path)[1])
                print "Could not extract enough features from cropped images, switching to original image %s" % new_image_path
                image_path = new_image_path
                surf_value = 500

                des = np.asarray(surf_function(image_path, surf_value)[1])

        if completed:
            histogram = build_histogram(km.labels_)

            X.append(histogram)
            y.append(image_id)
        else:
            print "Could not extract enough features from image %s for processing" % image_path

    return X, y


def build_histogram(data):
    histograms = {}
    for element in data:
        if element not in histograms:
            histograms[element] = 1
        else:
            histograms[element] += 1
    return histograms


def build_cluster_histograms(db):
    histogram = build_histogram(db.labels_)

    if -1 in histogram:
        del histogram[-1]

    try:
        sorted_clusters = sorted(histogram.iterkeys(), key=(lambda key: histogram[key]), reverse=True)
    except:
        sorted_clusters = None

    return sorted_clusters, histogram


def run_dbscan(coordinates, min_k):
    X = StandardScaler().fit_transform(coordinates)

    distance = 0.03
    min_samples = int(math.sqrt(len(coordinates))) * .5

    while True:
        db = DBSCAN(eps=distance, min_samples=min_samples).fit(X)

        clusters, histograms = build_cluster_histograms(db)

        if clusters and clusters[0] is not None:
            if (len(set(db.labels_)) > 3 and sum(x > -1 for x in db.labels_) > min_k) or histograms[clusters[0]] == len(coordinates):
                return db

        distance *= 1.25
