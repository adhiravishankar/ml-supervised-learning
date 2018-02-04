from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier


def linearsvc(x_train, x_test, y_train):
    print("Started Linear SVM")
    model = LinearSVC()
    model.fit(x_train, y_train)
    return model.predict(x_test)


def svm(x_train, x_test, y_train):
    print("Started SVM")
    model = SVC()
    model.fit(x_train, y_train)
    return model.predict(x_test)


def neural(x_train, x_test, y_train):
    print("Started Neural Network")
    model = MLPClassifier()
    model.fit(x_train, y_train)
    return model.predict(x_test)


def knn(x_train, x_test, y_train, neighbors):
    model = NearestNeighbors(n_neighbors=neighbors, metric='ball_tree')
    model.fit(x_train, y_train)
    results = model.kneighbors(x_test)


def decision(x_train, x_test, y_train):
    print("Started Decision Tree")
    model = DecisionTreeClassifier()
    model.fit(x_train, y_train)
    return model.predict(x_test)


def boost(x_train, x_test, y_train):
    model = GradientBoostingClassifier()
    model.fit(x_train, y_train)
    return model.predict(x_test)