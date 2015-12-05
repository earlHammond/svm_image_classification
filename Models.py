
import numpy as np

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multiclass import OneVsOneClassifier

import Clustering


def train_gradient_boosting_classifier(X, y, k, kfold, vectorize_strategy):
    X = vectorize_strategy(X)
    clf = GradientBoostingClassifier(n_estimators=k)

    scores = []
    for train, test in kfold:
        score = clf.fit(X[train], y[train]).score(X[test], y[test])
        scores.append(score)

    return scores


def train_svm(X, y, kernel, C, kfold, vectorize_strategy):
    X = vectorize_strategy(X)
    clf = SVC(kernel=kernel, C=C)

    scores = []
    for train, test in kfold:
        score = clf.fit(X[train], y[train]).score(X[test], y[test])
        scores.append(score)

    return scores


def train_forest(X, y, k, kfold, vectorize_strategy):
    X = vectorize_strategy(X)
    clf = RandomForestClassifier(n_estimators=k)

    scores = []
    for train, test in kfold:
        score = clf.fit(X[train], y[train]).score(X[test], y[test])
        scores.append(score)

    return scores


def merge_and_vectorize_features(X):
    vectorizer = DictVectorizer()
    X = merge_histograms(X)
    return vectorizer.fit_transform(X).todense()


def vectorize_multihistograms(X):
    vectorizer = DictVectorizer()
    holder = []
    for x in X:
        holder.append(vectorizer.fit_transform(x).todense())

    return np.vstack(holder).todense()


def merge_histograms(image_data):
    merged_data = []
    for histograms in image_data:
        merged_histogram = Clustering.get_base_histogram(len(image_data[0][0].keys()))
        for histogram_bin in merged_histogram:
            for histogram in histograms:
                merged_histogram[histogram_bin] += histogram[histogram_bin]
        merged_data.append(merged_histogram)

    return merged_data
