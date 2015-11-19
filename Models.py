
from sklearn.svm import SVC
from sklearn.feature_extraction import DictVectorizer


def train_svm(X, y, kernel, C, gamma):
    vectorizer = DictVectorizer()
    X = vectorizer.fit_transform(X)
    svm = SVC(kernel=kernel, C=C, gamma=gamma)
    svm.fit(X, y)
    return svm


def predict_model(X, y, model):
    pass
