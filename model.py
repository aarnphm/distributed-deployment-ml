from tensorflow.keras.layers import Activation, Dense, Dropout, Embedding, LSTM
from tensorflow.keras.models import Sequential

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import BertPreTrainedModel, BertModel


class TransformersBert(BertPreTrainedModel):
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


# TF BiLSTM tfmodel.
def TFLSTM(max_seq_len, vocab_size):
    model = Sequential()
    model.add(Embedding(vocab_size, 64, input_length=max_seq_len))
    model.add(TFLSTM(64, return_sequences=True))
    model.add(Dropout(0.5))
    model.add(TFLSTM(64))
    model.add(Dense(256, name='fc1'))
    model.add(Dropout(0.5))
    model.add(Dense(1, name='out'))
    model.add(Activation('sigmoid'))
    return model
