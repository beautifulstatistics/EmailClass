import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import chi2

from ALBERT_preprocess import load_corpus_and_labels

CORPUS, LABELS = load_corpus_and_labels()

print("\nmethod 1\n")
vectorizer = CountVectorizer(ngram_range=(1,2), stop_words= 'english')
X = vectorizer.fit_transform(CORPUS)
y = np.array(LABELS)

chi2score = chi2(X, y)

indices = np.argsort(chi2score[0])[::-1]

for i in indices[:10]:
    print(vectorizer.get_feature_names_out()[i])

print('\nmethod 2:\n')
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

corpus_train, corpus_test, labels_train, labels_test = train_test_split(CORPUS, LABELS, test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer(ngram_range=(1,2), stop_words='english')
X_train = vectorizer.fit_transform(corpus_train)
y_train = np.array(labels_train)

model = LogisticRegression()
model.fit(X_train, y_train)

X_test = vectorizer.transform(corpus_test)
y_test_pred = model.predict(X_test)

accuracy = accuracy_score(labels_test, y_test_pred)
print("\nTest accuracy: ", accuracy,'\n')

vectorizer_full = TfidfVectorizer(ngram_range=(1,2), stop_words='english')
X_full = vectorizer_full.fit_transform(CORPUS)
y_full = np.array(LABELS)

model_full = LogisticRegression()
model_full.fit(X_full, y_full)

feature_names_full = np.array(vectorizer_full.get_feature_names_out())
sorted_coef_index_full = model_full.coef_[0].argsort()

for index in sorted_coef_index_full[:-11:-1]:
    print(feature_names_full[index])

