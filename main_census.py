from sklearn.metrics import confusion_matrix, accuracy_score

from classifiers import neural, decision, boost, linearsvc
from preprocess_census import read_data

print("Started pre-processing data")
X_train, X_test, Y_train, Y_test = read_data()
print("Finished pre-processing data")

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