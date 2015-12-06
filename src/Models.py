from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier


def train_gradient_boosting_classifier(X, y, k, kfold, vectorize_strategy):
    print "Training Boost Model"

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
    print "Training Random Forest Model"

    X = vectorize_strategy(X)
    clf = RandomForestClassifier(n_estimators=k)

    scores = []
    for train, test in kfold:
        score = clf.fit(X[train], y[train]).score(X[test], y[test])
        scores.append(score)

    return scores
