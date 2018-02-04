from collections import Counter
from email.parser import BytesParser
from email.policy import default

import nltk
import numpy
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split


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
        email = (hs, hsnum, message, words_in_message)
        emails.append(email)
    return all_words_counter, emails


def extract_features(all_words, emails):
    most_common_words = all_words.most_common(3000)
    features_matrix = numpy.zeros([len(emails), len(most_common_words)], dtype=int)
    train_labels = numpy.zeros(len(emails), dtype=int)
    for emailId, email in enumerate(emails):
        for word in email[3]:
            for index, dicWord in enumerate(most_common_words):
                if dicWord[0] == word:
                    wordID = index
                    features_matrix[emailId, wordID] = email[3].count(word)
        train_labels[emailId] = email[1]
    return train_test_split(features_matrix, train_labels, test_size=0.4)