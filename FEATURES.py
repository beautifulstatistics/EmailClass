import os
from joblib import dump
import numpy as np

from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

from get_data import load_features_and_labels


X, LABELS = load_features_and_labels()

X = np.array(X).reshape(-1,1)

lr = LogisticRegression()
scores = cross_val_score(lr, X, LABELS, cv=5)

print('Accuracy: ', scores.mean())

lr.fit(X, LABELS)

path = os.path.join('models','FEATURES','lrmodel.joblib')
dump(lr, path)