import Kernels

from numpy import mean
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import MultinomialNB


def train_naive_bayes(X, y, kfold, vectorization_strategy):
    print "Training Naive Bayes Model"

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


def train_svm(X, y, kfold, vectorization_strategy, all_scores):
    print "Training SVM Model"
    kernels = ['poly', 'rbf', 'sigmoid', Kernels.laplacian_kernel]
    penalties = [0.01, 1, 100, 1000, 10000]
    X = vectorization_strategy(X)

    for kernel in kernels:
        for penalty in penalties:
            clf = SVC(kernel=kernel, C=penalty)

            scores = []
            for train, test in kfold:
                score = clf.fit(X[train], y[train]).score(X[test], y[test])
                scores.append(score)
            scores.append(mean(scores))
            all_scores["%s_%s_%i" % ('SVM', str(kernel), penalty)] = scores

    return all_scores


def train_forest(X, y, kfold, vectorization_strategy):
    print "Training Random Forest Model"

    X = vectorization_strategy(X)
    clf = RandomForestClassifier(n_estimators=20)

    scores = []
    for train, test in kfold:
        score = clf.fit(X[train], y[train]).score(X[test], y[test])
        scores.append(score)

    return scores
