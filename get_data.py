import os
import random

def load_emails_from_directory(path, label):
    corpus = []
    labels = []
    
    for file_name in os.listdir(path):
        file_path = os.path.join(path, file_name)
        with open(file_path, 'rb') as file:
            content = file.read().decode('utf-8', errors='replace')
            labels.append(label)
            corpus.append(content)

    return corpus, labels

def load_corpus_and_labels():
    corpus = []
    labels = []

    spam_path = os.path.join('data', 'spam')
    spam_corpus, spam_labels = load_emails_from_directory(spam_path, 1)
    corpus.extend(spam_corpus)
    labels.extend(spam_labels)

    not_spam_path = os.path.join('data', 'not_spam')
    not_spam_corpus, not_spam_labels = load_emails_from_directory(not_spam_path, 0)
    corpus.extend(not_spam_corpus)
    labels.extend(not_spam_labels)

    return corpus, labels

def load_features_and_labels():
    spam_path = os.path.join('data', 'spam')
    not_spam_path = os.path.join('data', 'not_spam')
    
    Ns = len(os.listdir(spam_path))
    Nns = len(os.listdir(not_spam_path))
              
    features = [random.choice([0,23,445,32,23]) for _ in range(Ns + Nns)]
    
    return features, [0] * Ns + [1] * Nns