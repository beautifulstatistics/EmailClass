from transformers import AlbertTokenizer, Trainer, TrainingArguments
from get_data import load_corpus_extra_and_labels
from ALBERTExtra import AlbertExtra
from datasets import Dataset
from torch.utils.data import random_split
import torch

from sklearn.metrics import accuracy_score, precision_recall_fscore_support

tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')

def tokenize_function(examples):
    return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=512)

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

CORPUS, LABELS, EXTRA_FEATURES = load_corpus_extra_and_labels()

raw_datasets = Dataset.from_dict({'text': CORPUS, 'label_names': LABELS, 'extra_features': EXTRA_FEATURES})
tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)

train_size = int(0.8 * len(tokenized_datasets))
val_size = len(tokenized_datasets) - train_size
train_dataset, val_dataset = random_split(tokenized_datasets, [train_size, val_size])

feature_dim = len(EXTRA_FEATURES[0])
num_labels = len(set(LABELS))

model = AlbertExtra(feature_dim, num_labels)

training_args = TrainingArguments(
    output_dir='results',
    num_train_epochs=2,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='logs',
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)

trainer.train()

# eval_result = trainer.evaluate()

# print(eval_result)

# # Now train on entire dataset

# full_dataset = tokenized_datasets

# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=full_dataset
# )

# trainer.train()

# # To save the model
# torch.save(model.state_dict(), 'models/albert_plus.pt')
# tokenizer.save_pretrained('models')
