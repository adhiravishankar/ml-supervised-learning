from collections import Counter
from email.parser import BytesParser
from email.policy import default

import nltk
import numpy
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split


def get_data(test_size=0.4):
    # all_words, emails = extract_emails()
    # features_matrix, train_labels = extract_features(all_words, emails)
    # numpy.save('email_features.npy', features_matrix)
    # numpy.save('email_labels.npy', train_labels)
    features_matrix = numpy.load('email_features.npy')
    train_labels = numpy.load('email_labels.npy')
    X_train, X_test, Y_train, Y_test = train_test_split(features_matrix, train_labels, test_size=test_size)
    return features_matrix, train_labels, X_train, X_test, Y_train, Y_test

def extract_emails():
    nltk.download('stopwords')
    stopWords = set(stopwords.words('english'))
    all_words_counter = Counter()
    emails = []
    file06 = open('06.txt')
    for line in file06:
        line = line.rstrip()
        hs, file = line.split(' ')
        emailfile = open(file, 'rb')
        message = BytesParser(policy=default).parse(emailfile)
        words_in_message = []
        if isinstance(message._payload, str):
            message_text = message._payload.lower().replace('\n', ' ').replace('\t', ' ').replace('"', ' ')
            message_text = nltk.re.sub("[!=|@#$%^&*(){};:,./<>?_+-]", " ", message_text)
            message_text = nltk.re.sub(' +', ' ', message_text)
            words_in_message = [x for x in message_text.split(' ') if x not in stopWords]
            all_words_counter.update(words_in_message)
        hsnum = 0
        if hs == 'spam':
            hsnum = 1
        email = (hs, hsnum, words_in_message)
        emails.append(email)
        if len(emails) % 1000 == 0:
            print("Processed ", len(emails))
    fivet_common_words = [a for a, b in all_words_counter.most_common(5000)]
    return fivet_common_words, emails


def extract_features(all_words, emails):
    features_matrix = numpy.zeros([len(emails), len(all_words)], dtype=int)
    train_labels = numpy.zeros(len(emails), dtype=int)
    for emailId, email in enumerate(emails):
        for word in set(email[2]):
            if word in all_words:
                wordID = all_words.index(word)
                features_matrix[emailId, wordID] = email[2].count(word)
        train_labels[emailId] = email[1]
        if emailId % 1000 == 0:
            print("Extracted from ", emailId)
    return features_matrix, train_labels