from preprocess_email import extract_emails, extract_features


all_words, emails = extract_emails()
X_train, X_test, Y_train, Y_test = extract_features(all_words, emails)


