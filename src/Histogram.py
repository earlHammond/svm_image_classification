import os

import numpy as np

import ImageProcessing
import Clustering


def build_image_histograms(surf_spread, k, cluster_centers, descriptors_lookup):
    X = []
    y = []
    km = Clustering.get_kmeans_cluster(k, cluster_centers)

    for image_path in descriptors_lookup:
        print "Building Histogram for %s" % image_path
        image_id = os.path.split(os.path.dirname(image_path))[1]
        img = ImageProcessing.load_image(image_path)
        sections = ImageProcessing.divide_image(img)

        if surf_spread == 'sparse':
            sections = [image_path]

        histograms = []
        for section in sections:
            kp, des = ImageProcessing.run_surf(section, surf_spread)
            surf_value = 400
            attempts = 0
            completed = False

            while attempts < 10 and not completed:
                try:
                    descriptors = np.vstack(des)
                    km.fit(descriptors)
                    completed = True
                except:
                    print "Re-running %s with SURF value set to: %i" % (image_path, surf_value)
                    surf_value *= 0.80
                    attempts += 1
                    des = ImageProcessing.run_surf(section, surf_spread)[1]

            if completed:
                histogram = build_histogram(km.labels_, k)
                histograms.append(histogram)
            else:
                print "Could not extract enough features from image %s for processing" % image_path
                histogram = build_histogram([], k)
                histograms.append(histogram)

        if len(histograms) == 4 or (surf_spread == 'sparse' and len(histograms) == 1):
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
    for x in xrange(0, k):
        histogram[x] = 0
    return histogram
