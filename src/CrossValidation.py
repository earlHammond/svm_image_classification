from collections import Counter

from sklearn.cross_validation import KFold, StratifiedKFold


def get_kfold(y, n_folds=5):
    counts = Counter(y)
    if n_folds > min(counts.values()):
        n_folds = min(counts.values())

    return KFold(len(y), n_folds=n_folds)


def get_stratified_fold(y, n_folds=5):
    counts = Counter(y)
    if n_folds > min(counts.values()):
        n_folds = min(counts.values())

    return StratifiedKFold(y, n_folds=n_folds)
