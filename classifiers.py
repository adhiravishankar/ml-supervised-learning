from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier


def svc(x_train, x_test, y_train):
    model = LinearSVC()
    model.fit(x_train, y_train)
    return model.predict(x_test)


def neural(x_train, x_test, y_train):
    model = MLPClassifier()
    model.fit(x_train, y_train)
    return model.predict(x_test)


def knn(x_train, x_test, y_train):
    model = NearestNeighbors(n_neighbors=3, metric='ball_tree')
    model.fit(x_train, y_train)
    results = model.kneighbors(x_test)


def decision(x_train, x_test, y_train):
    model = DecisionTreeClassifier()
    model.fit(x_train, y_train)
    return model.predict(x_test)


def boost(x_train, x_test, y_train):
    model = GradientBoostingClassifier()
    model.fit(x_train, x_test)
    return model.predict(x_test)