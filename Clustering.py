import os
import math
import numpy as np

from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler


def cluster_key_points(descriptors, k):
    print "Clustering Image Features to %s clusters" % k
    km = KMeans(n_clusters=k, max_iter=200)
    descriptors = np.vstack(descriptors.values())
    km.fit(descriptors)
    return km.cluster_centers_


def build_histograms_from_features(k, cluster_centers, descriptors_lookup):
    km = KMeans(n_clusters=k, max_iter=200, init=cluster_centers, n_init=1)

    X = []
    y = []
    for image_path in descriptors_lookup:
        print "Building Histogram for %s" % image_path
        whale_id = os.path.split(os.path.dirname(image_path))[1]
        descriptors = np.vstack(descriptors_lookup[image_path])
        km.fit(descriptors)
        histogram = build_histogram(km.labels_)

        X.append(histogram)
        y.append(whale_id)

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


def run_dbscan(coordinates):
    X = StandardScaler().fit_transform(coordinates)

    distance = 0.03
    min_samples = int(math.sqrt(len(coordinates))) * .5

    while True:
        db = DBSCAN(eps=distance, min_samples=min_samples).fit(X)

        clusters, histograms = build_cluster_histograms(db)

        if clusters and clusters[0] is not None:
            if len(set(db.labels_)) > 3 or histograms[clusters[0]] == len(coordinates):
                return db

        distance *= 1.25
