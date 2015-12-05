
import random


from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer


def train_svm(X, y, kernel, C, kfold):
    X = vectorize_features(X)
    svc = SVC(kernel=kernel, C=C, probability=True)

    scores = []
    for train, test in kfold:
        score = svc.fit(X[train], y[train]).score(X[test], y[test])
        scores.append(score)

    return scores


def train_forest(X, y, k, kfold):
    X = vectorize_features(X)
    clf = RandomForestClassifier(n_estimators=k)

    scores = []
    for train, test in kfold:
        score = clf.fit(X[train].todense(), y[train]).score(X[test].todense(), y[test])
        scores.append(score)

    return scores


def vectorize_features(X):
    vectorizer = DictVectorizer()
    return vectorizer.fit_transform(X)


def custom_k_folds(y, n_folds=5, min_response=1):
    random.seed(12345)
    response_values = {}

    for response in set(y):
        index_count = [i for i, x in enumerate(y) if x == response]
        if index_count and len(index_count) > 1:
            response_values[response] = index_count

    folds = []
    run = 0
    while run < n_folds:
        train = []
        test = []

        for value in response_values:
            response_indices = response_values[value]
            test_response_indices = []

            while len(test_response_indices) < min_response:
                test_index = random.choice(response_indices)
                if test_index not in test_response_indices:
                    test_response_indices.append(test_index)
                    test.append(test_index)

            for index in response_indices:
                if index not in test:
                    train.append(index)

        folds.append((train, test))
        run += 1

    return folds
