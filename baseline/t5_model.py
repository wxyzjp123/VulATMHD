import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

class RobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""
    def __init__(self, config, num_labels):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.out_proj = nn.Linear(config.hidden_size, num_labels)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

class Model(nn.Module):   
    def __init__(self, decoder, config, tokenizer, args, num_labels):
        super(Model, self).__init__()
        self.decoder = decoder
        # self.score = nn.Linear(config.n_embd, num_labels, bias=False)
        self.classifier = RobertaClassificationHead(config, num_labels)
        self.tokenizer = tokenizer
        self.args = args
    
    def forward(self, input_ids, labels=None, logit_adjustment=None, focal_loss=False):

        last_hidden_state = self.decoder.encoder(input_ids, attention_mask=input_ids.ne(self.tokenizer.pad_token_id)).last_hidden_state
        logits = self.classifier(last_hidden_state)

        prob = torch.softmax(logits, dim=-1)
        if labels is not None:
            if logit_adjustment is not None:
                logits = logits + logit_adjustment
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits, labels)
            return loss, prob
        else:
            return prob