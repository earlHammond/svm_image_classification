import os
import math
import numpy as np
import cv2

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

        img = cv2.imread(image_path, cv2.CV_LOAD_IMAGE_GRAYSCALE)
        sections = ImageProcessing.divide_image(img, gray=True)
        histograms = []
        for section in sections:
            kp, des = surf_function(section)
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

            if completed:
                histogram = build_histogram(km.labels_, k)
                histograms.append(histogram)
            else:
                print "Could not extract enough features from image %s for processing" % image_path
                break

        if len(histograms) == 4:
            X.append(histograms)
            y.append(image_id)

    return X, y


def build_histogram(data, k):
    histograms = get_base_histogram(k)
    for element in data:
        histograms[element] += 1
    return histograms


def get_base_histogram(k):
    histogram = {}
    for x in xrange(0,k):
        histogram[x] = 0
    return histogram


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
