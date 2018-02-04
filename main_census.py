from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

from classifiers import neural, decision, boost, linearsvc, plot_learning_curve, plot_learning_curves
from preprocess_census import read_data

print("Started pre-processing data")
X, Y, X_train, X_test, Y_train, Y_test = read_data()
print("Finished pre-processing data")

plot_learning_curves(X, Y)

results3 = decision(X_train, X_test, Y_train)
print("Decision: ", confusion_matrix(Y_test, results3))
print("Accuracy of Decision: ", accuracy_score(Y_test, results3))

results4 = boost(X_train, X_test, Y_train)
print("Boost: ", confusion_matrix(Y_test, results4))
print("Accuracy of Boost: ", accuracy_score(Y_test, results4))

results5 = linearsvc(X_train, X_test, Y_train)
print("Linear SVM: ", confusion_matrix(Y_test, results5))
print("Accuracy of Linear SVM: ", accuracy_score(Y_test, results5))

# results1 = svm(X_train, X_test, Y_train)
# print("SVM: ", confusion_matrix(Y_test, results1))
# print("Accuracy of SVM: ", accuracy_score(Y_test, results1))

results2 = neural(X_train, X_test, Y_train)
print("Neural: ", confusion_matrix(Y_test, results2))
print("Accuracy of Neural: ", accuracy_score(Y_test, results2))