import torch.nn as nn
from transformers import BertModel
import torch.nn.functional as F
import torch

class BertcnnModel(nn.Module):
    def __init__(self, hidden_size=768, filter_sizes=(2, 3, 4), num_filters=256, n_classes=2):
        super(BertcnnModel, self).__init__()
        self.hidden_size = hidden_size
        self.filter_sizes = filter_sizes                    # 卷积核尺寸
        self.num_filters = num_filters                      # 卷积核数量(channels数)
        self.n_classes = n_classes                          # 类别

        self.bert = BertModel.from_pretrained("bert-base-chinese")
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, num_filters, kernel_size=(k, hidden_size)) for k in filter_sizes])
        self.fc_cnn = nn.Sequential(
            nn.Linear(num_filters * len(filter_sizes), 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
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
    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, input_ids, attention_mask):
        encoder_out, text_cls = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=False
        )
        print(encoder_out.shape)
        out = encoder_out.unsqueeze(1)
        out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1)
        out = self.fc_cnn(out)
        return out
