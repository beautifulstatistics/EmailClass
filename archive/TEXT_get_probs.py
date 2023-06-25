from transformers import AlbertTokenizer, AlbertForSequenceClassification
import torch

def get_probabilities(model, text):
    inputs = tokenizer(text, padding='max_length', truncation=True, max_length=512, return_tensors='pt')
    outputs = model(**inputs)
    probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
    return [x[1].item() for x in probabilities]


model = AlbertForSequenceClassification.from_pretrained('models/TEXT')
tokenizer = AlbertTokenizer.from_pretrained('models/TEXT')