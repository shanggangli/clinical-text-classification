import torch.nn as nn
from transformers import BertModel
import torch.nn.functional as F

class bertModel(nn.Module):
    def __init__(self, n_classes):
        super(bertModel, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-chinese")
        self.drop = nn.Dropout(p=0.3)
        self.fc = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 8),
            nn.ReLU(inplace=True),
            nn.Linear(8, n_classes),
            nn.ReLU(inplace=True),
        )

    def forward(self, input_ids, attention_mask):
        _, pooled_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict = False
        )
        out = self.fc(pooled_output)
        return out
