import os

def load_emails_from_directory(path, label, corpus, labels):
    for file_name in os.listdir(path):
        file_path = os.path.join(path, file_name)
        with open(file_path, 'rb') as file:
            content = file.read().decode('utf-8', errors='replace')
            labels.append(label)
            corpus.append(content)

def load_corpus_and_labels():
    corpus = []
    labels = []

    spam_path = os.path.join('data', 'spam')
    load_emails_from_directory(spam_path, 1, corpus, labels)

    not_spam_path = os.path.join('data', 'not_spam')
    load_emails_from_directory(not_spam_path, 0, corpus, labels)

    return corpus, labels