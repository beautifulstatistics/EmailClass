import torch
from torch import nn
from transformers import AlbertModel

class AlbertExtra(nn.Module):
    def __init__(self, feature_dim, num_labels):
        super().__init__()
        
        self.albert = AlbertModel.from_pretrained('albert-base-v2')
        self.feature_processor = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
        )
        
        self.classifier = nn.Linear(self.albert.config.hidden_size + 64, num_labels)

def forward(self, input_ids=None, attention_mask=None, extra_features=None, label_names=None):

    extra_features = extra_features.float()
    
    outputs = self.albert(input_ids=input_ids, attention_mask=attention_mask)
    albert_output = outputs[0]
    
    processed_features = self.feature_processor(extra_features)
    
    combined = torch.cat((albert_output, processed_features), dim=1)
    
    logits = self.classifier(combined)

    loss = None
    if label_names is not None:
        if self.num_labels == 1:
            loss_fct = torch.nn.BCEWithLogitsLoss()
            loss = loss_fct(logits.view(-1), label_names.view(-1))
        else:
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), label_names.view(-1))

    output = (logits,) + outputs[2:]
    return ((loss,) + output) if loss is not None else output

