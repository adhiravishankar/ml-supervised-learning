import time
from sklearn.metrics import confusion_matrix, accuracy_score
from classifiers import neural, decision, boost, linearsvc, knn, plot_learning_curves
from preprocess_email import get_data

X, Y, X_train, X_test, Y_train, Y_test = get_data()

plot_learning_curves(X, Y)

svc_start = time.time()
results1 = linearsvc(X_train, X_test, Y_train)
svc_end = time.time()
print("SVM: ", confusion_matrix(Y_test, results1))
print("Accuracy of SVM: ", accuracy_score(Y_test, results1))
print("Time for SVM: ", (svc_end - svc_start), " s")

neural_start = time.time()
results2 = neural(X_train, X_test, Y_train)
neural_stop = time.time()
print("Neural: ", confusion_matrix(Y_test, results2))
print("Accuracy of Neural: ", accuracy_score(Y_test, results2))
print("Time for Neural: ", (neural_stop - neural_start), " s")

decision_start = time.time()
results3 = decision(X_train, X_test, Y_train)
decision_end = time.time()
print("Decision: ", confusion_matrix(Y_test, results3))
print("Accuracy of Decision: ", accuracy_score(Y_test, results3))
print("Time for Decision: ", (decision_end - decision_start), " s")

boost_start = time.time()
results4 = boost(X_train, X_test, Y_train)
boost_end = time.time()
print("Boost: ", confusion_matrix(Y_test, results4))
print("Accuracy of Boost: ", accuracy_score(Y_test, results4))
print("Time for Boost: ", (boost_end - boost_start), " s")

knn_start = time.time()
results5 = knn(X_train, X_test, Y_train, 3)
knn_end = time.time()
print("KNN (k=3): ", confusion_matrix(Y_test, results5))
print("Accuracy of KNN (k=3): ", confusion_matrix(Y_test, results5))
print("Time for KNN (k=3): ", (knn_end - knn_start), " s")

knn5_start = time.time()
results6 = knn(X_train, X_test, Y_train)
knn5_end = time.time()
print("KNN (k=5): ", confusion_matrix(Y_test, results6))
print("Accuracy of KNN (k=5): ", confusion_matrix(Y_test, results6))
print("Time for KNN (k=5): ", (knn5_end - knn5_start), " s")
