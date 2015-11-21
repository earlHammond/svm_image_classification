
import random

import Constants

from sklearn.svm import SVC
from sklearn.feature_extraction import DictVectorizer


def train_svm(X, y, kernel, C):
    X = vectorize_features(X)
    svc = SVC(kernel=kernel, C=C, probability=True)

    kfold = custom_k_folds(y, n_folds=3)
    scores = []
    for train, test in kfold:
        score = svc.fit(X[train], y[train]).score(X[test], y[test])
        scores.append(score)

    return scores


def vectorize_features(X):
    vectorizer = DictVectorizer()
    return vectorizer.fit_transform(X)


def custom_k_folds(y, n_folds=5):
    training_indices = {}

    for whale in Constants.WHALES_WITH_MULTIPLE_RECORDS:
        whale_images = [i for i, x in enumerate(y) if x == whale]
        if whale_images:
            training_indices[whale] = whale_images

    folds = []
    run = 0
    while run < n_folds:
        train = []
        test = []

        for whale in training_indices:
            whale_indices = training_indices[whale]
            test_index = random.choice(whale_indices)
            test.append(test_index)
            for index in whale_indices:
                if index != test_index:
                    train.append(index)

        folds.append((train, test))
        run += 1

    return folds
