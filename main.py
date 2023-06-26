import os
from joblib import load
import numpy as np

from transformers import AlbertTokenizer, AlbertForSequenceClassification
import torch

path = os.path.join('models','TEXT')
text_model = AlbertForSequenceClassification.from_pretrained(path)
tokenizer = AlbertTokenizer.from_pretrained(path)

path = os.path.join('models','FEATURES','lrmodel.joblib')
features_model = load(path)

def get_probabilities_from_text(model, text):
    inputs = tokenizer(text, padding='max_length', truncation=True, max_length=512, return_tensors='pt')
    outputs = model(**inputs)
    tensors = torch.nn.functional.softmax(outputs.logits, dim=-1)
    return [p[1].item() for p in tensors]

def get_probabilities_from_features(model, features):
    return [p[1] for p in model.predict_proba(features)]


######################

new_text = ['I HAVE AN OFFER FOR YOU!','Dear Maria,']
new_features = np.array([26,3]).reshape(-1,1)

text_probs = get_probabilities_from_text(text_model, new_text)
features_probs = get_probabilities_from_features(features_model, new_features)

final_probs = [(p1 + p2)/2 for p1,p2 in zip(text_probs, features_probs)]

print(final_probs)
