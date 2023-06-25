from transformers import AlbertTokenizer, AlbertForSequenceClassification
import torch

model = AlbertForSequenceClassification.from_pretrained('models/TEXT')
tokenizer = AlbertTokenizer.from_pretrained('models/TEXT')

def get_probabilities(model, text):
    inputs = tokenizer(text, padding='max_length', truncation=True, max_length=512, return_tensors='pt')
    outputs = model(**inputs)
    tensors = torch.nn.functional.softmax(outputs.logits, dim=-1)
    return [p[1].item() for p in tensors]
