from transformers import AlbertTokenizer, AlbertForSequenceClassification
import torch

def get_probabilities(model, text):
    inputs = tokenizer(text, padding='max_length', truncation=True, max_length=512, return_tensors='pt')
    outputs = model(**inputs)
    probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
    return probabilities


model = AlbertForSequenceClassification.from_pretrained('models')
tokenizer = AlbertTokenizer.from_pretrained('models')

text = ["I HAVE AN OFFER FOR YOU!",'Dear Maria,']
probabilities = get_probabilities(model, text)

print("Probabilities:", probabilities)
