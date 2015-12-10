import math

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import MultinomialNB


def train_naive_bayes(X, y, kfold, vectorization_strategy):
    print "Training Naive Bayes"

    X = vectorization_strategy(X)
    clf = MultinomialNB()

    scores = []
    for train, test in kfold:
        score = clf.fit(X[train], y[train]).score(X[test], y[test])
        scores.append(score)

    return scores


def train_gradient_boosting_classifier(X, y, kfold, vectorization_strategy):
    print "Training Boost Model"

    X = vectorization_strategy(X)
    clf = GradientBoostingClassifier(n_estimators=100)

    scores = []
    for train, test in kfold:
        score = clf.fit(X[train], y[train]).score(X[test], y[test])
        scores.append(score)

    return scores


def train_svm(X, y, kernel, C, kfold, vectorization_strategy):

    X = vectorization_strategy(X)
    clf = SVC(kernel=kernel, C=C)

    scores = []
    for train, test in kfold:
        score = clf.fit(X[train], y[train]).score(X[test], y[test])
        scores.append(score)

    return scores


def train_forest(X, y, kfold, vectorization_strategy):
    print "Training Random Forest Model"

    X = vectorization_strategy(X)
    clf = RandomForestClassifier(n_estimators=20)

    scores = []
    for train, test in kfold:
        score = clf.fit(X[train], y[train]).score(X[test], y[test])
        scores.append(score)

    return scores
