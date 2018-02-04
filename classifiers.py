import numpy
import matplotlib.pyplot as mplot
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import learning_curve
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
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


def knn(x_train, x_test, y_train, neighbors=5):
    model = KNeighborsClassifier(n_neighbors=neighbors)
    model.fit(x_train, y_train)
    return model.predict(x_test)


def decision(x_train, x_test, y_train):
    print("Started Decision Tree")
    model = DecisionTreeClassifier()
    model.fit(x_train, y_train)
    return model.predict(x_test)


def boost(x_train, x_test, y_train):
    model = GradientBoostingClassifier()
    model.fit(x_train, y_train)
    return model.predict(x_test)


def plot_learning_curves(X, Y):
    #mplot1 = plot_learning_curve(DecisionTreeClassifier(), "Decision Tree", X, Y)
    #mplot1.show()

    #mplot2 = plot_learning_curve(GradientBoostingClassifier(), "Boosted Decision Tree", X, Y)
    #mplot2.show()

    #mplot3 = plot_learning_curve(LinearSVC(), "Linear SVC", X, Y)
    #mplot3.show()

    #mplot4 = plot_learning_curve(MLPClassifier(), "Neural Network", X, Y)
    #mplot4.show()

    # mplot5 = plot_learning_curve(KNeighborsClassifier(), "K-Nearest Neighbors (k=5)", X, Y)
    # mplot5.show()

    mplot6 = plot_learning_curve(KNeighborsClassifier(n_neighbors=3), "K-Nearest Neighbors (k=3)", X, Y)
    mplot6.show()


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None, n_jobs=1, train_sizes=numpy.linspace(.1, 1.0, 5)):
    """
    Generate a simple plot of the test and traning learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : integer, cross-validation generator, optional
        If an integer is passed, it is the number of folds (defaults to 3).
        Specific cross-validation objects can be passed, see
        sklearn.cross_validation module for the list of possible objects

    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).
    """
    mplot.figure()
    mplot.title(title)
    if ylim is not None:
        mplot.ylim(*ylim)
    mplot.xlabel("Training examples")
    mplot.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = numpy.mean(train_scores, axis=1)
    train_scores_std = numpy.std(train_scores, axis=1)
    test_scores_mean = numpy.mean(test_scores, axis=1)
    test_scores_std = numpy.std(test_scores, axis=1)
    mplot.grid()

    mplot.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color="r")
    mplot.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color="g")
    mplot.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    mplot.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")

    mplot.legend(loc="best")
    return mplot