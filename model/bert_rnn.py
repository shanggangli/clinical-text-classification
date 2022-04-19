import torch.nn as nn
from transformers import BertModel


class BertrnnModel(nn.Module):
    def __init__(self, hidden_size=768, rnn_hidden = 768, dropout=0.3,num_layers=2,n_classes=2):
        super(BertrnnModel, self).__init__()
        self.hidden_size = hidden_size
        self.rnn_hidden = rnn_hidden  #
        self.n_classes = n_classes  # 类别

        self.bert = BertModel.from_pretrained("bert-base-chinese")
        self.num_layers = num_layers
        self.lstm = nn.LSTM(hidden_size, rnn_hidden, num_layers,
                            bidirectional=True, batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc_rnn = nn.Linear(rnn_hidden * 2, n_classes)

    def forward(self, input_ids, attention_mask):
        encoder_out, text_cls = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=False
        )
        out, _ = self.lstm(encoder_out)
        out = self.dropout(out)
        out = self.fc_rnn(out[:, -1, :])  # 句子最后时刻的 hidden state
        return out
