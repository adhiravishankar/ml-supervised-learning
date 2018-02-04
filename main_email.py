from sklearn.metrics import confusion_matrix, accuracy_score
from classifiers import neural, decision, boost, linearsvc
from preprocess_email import extract_emails, extract_features


all_words, emails = extract_emails()
X_train, X_test, Y_train, Y_test = extract_features(all_words, emails)

results1 = linearsvc(X_train, X_test, Y_train)
results2 = neural(X_train, X_test, Y_train)
results3 = decision(X_train, X_test, Y_train)
results4 = boost(X_train, X_test, Y_train)

print("SVC: ", confusion_matrix(Y_test, results1))
print("Neural: ", confusion_matrix(Y_test, results2))
print("Decision: ", confusion_matrix(Y_test, results3))
print("Boost: ", confusion_matrix(Y_test, results4))

print("Accuracy of SVC: ", accuracy_score(Y_test, results1))
print("Accuracy of Neural: ", accuracy_score(Y_test, results2))
print("Accuracy of Decision: ", accuracy_score(Y_test, results3))
print("Accuracy of Boost: ", accuracy_score(Y_test, results4))


