from sklearn.cross_validation import KFold, StratifiedKFold


def get_kfold(y, n_folds=5):
    if n_folds < len(set(y)):
        n_folds = len(set(y))

    return KFold(len(y), n_folds=n_folds)


def get_stratified_fold(y, n_folds=5):
    if n_folds < len(set(y)):
        n_folds = len(set(y))

    return StratifiedKFold(y, n_folds=n_folds)
