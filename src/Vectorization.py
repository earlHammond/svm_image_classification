import numpy as np

import Histogram

from sklearn.feature_extraction import DictVectorizer


def merge_and_vectorize_features(X):
    vectorizer = DictVectorizer()
    X = merge_histograms(X)
    return vectorizer.fit_transform(X).todense()


def merge_histograms(image_data):
    merged_data = []
    for histograms in image_data:
        merged_histogram = Histogram.get_base_histogram(len(image_data[0][0].keys()))
        for histogram_bin in merged_histogram:
            for histogram in histograms:
                merged_histogram[histogram_bin] += histogram[histogram_bin]
        merged_data.append(merged_histogram)

    return merged_data


def vectorize_multi_histograms(X):
    vectorizer = DictVectorizer()
    holder = []
    for x in X:
        holder.append(vectorizer.fit_transform(x).todense())

    return np.vstack(holder)
