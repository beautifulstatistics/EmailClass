from sklearn.metrics import mean_squared_error
from datasets import Dataset

from transformers import AlbertTokenizer, AlbertForSequenceRegression, Trainer, TrainingArguments
from torch.utils.data import random_split

from get_data import load_corpus_and_labels

tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
model = AlbertForSequenceRegression.from_pretrained('albert-base-v2')

def tokenize_function(examples):
    return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=512)

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions
    mse = mean_squared_error(labels, preds)
    return {
        'mse': mse
    }

CORPUS, LABELS = load_corpus_and_labels()

raw_datasets = Dataset.from_dict({'text': CORPUS, 'labels': LABELS})
tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)

train_size = int(0.8 * len(tokenized_datasets))
val_size = len(tokenized_datasets) - train_size
train_dataset, val_dataset = random_split(tokenized_datasets, [train_size, val_size])

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
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.train()

eval_result = trainer.evaluate()

print(eval_result)

# Now train on entire dataset

full_dataset = tokenized_datasets

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=full_dataset
)
trainer.train()

trainer.save_model('models/TEXT')
tokenizer.save_pretrained('models/TEXT')
