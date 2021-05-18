from typing import List

import torch
import torch.nn as nn
from transformers import BertPreTrainedModel, BertModel, AutoTokenizer

from args import model_name_or_path
from manager import Manager


class BertForSentimentClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # The classification layer that takes the [CLS] representation and outputs the logit
        self.cls_layer = nn.Linear(config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        """
        :param input_ids: Tensor of shape [B, T] containing token ids of sequences
        :param attention_mask: Tensor of shape [B, T] containing attention masks to be used to avoid contribution of
               PAD tokens.
        :return: logits
        """
        # Feed the input to Bert model to obtain outputs
        outs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # Obtain the representations of [CLS] heads
        cls_reps = outs.last_hidden_state[:, 0]
        cls_reps = self.dropout(cls_reps)
        logits = self.cls_layer(cls_reps)
        return logits


class ManagedBertModel(Manager):
    def __init__(self):
        super(ManagedBertModel, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    def setup_model(self, *args, **kwargs):
        self.model = BertForSentimentClassification.from_pretrained(model_name_or_path)

    def predict(self, batch):
        with torch.no_grad():
            tokens = self.tokenizer.tokenize(batch)
            tokens = ['[CLS]'] + tokens + ['[SEP]']
            tokens_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            seq = torch.tensor(tokens_ids)
            seq = seq.unsqueeze(0)
            attn_mask = (seq != 0).long()
            logit = self.model(seq, attn_mask)
            prob = torch.sigmoid(logit.unsqueeze(-1))
            prob = prob.item()
            soft_prob = prob > 0.5
            if soft_prob == 1:
                return [int(prob*100)]
            else:
                return [int(100 - prob * 100)]
