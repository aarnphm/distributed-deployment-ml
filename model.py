from tensorflow.keras.layers import Activation, Dense, Dropout, Embedding, LSTM
from tensorflow.keras.models import Sequential

import torch
import torch.nn as nn

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


# PyTorch BiLSTM tfmodel.
class TorchLSTM(nn.Module):
    def __init__(
        self, input_dim, emb_dim, hid_dim, output_dim, n_layer, dropout, pad_idx
    ):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(
            emb_dim, hid_dim, num_layers=n_layer, bidirectional=True, dropout=dropout
        )
        self.fc = nn.Linear(2 * hid_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text, lengths):
        # [seq_len, batch_size, emb_dim]
        embedded = self.dropout(self.embedding(text))
        # https://discuss.pytorch.org/t/simple-working-example-how-to-use-packing-for-variable-length-sequence-inputs-for-rnn/2120
        packed_emb = nn.utils.rnn.pack_padded_sequence(
            embedded, lengths, enforce_sorted=False
        )
        packed_out, (hidden, _) = self.lstm(packed_emb)

        # outputs : [seq_len, batch_size, n_direction * hid_dim]
        # hid : [n_layers * n_direction, batch_size, hid_dim]
        out, _ = nn.utils.rnn.pad_packed_sequence(packed_out)

        # [batch_size, hid_dim]
        hidden_fwd, hidden_bck = hidden[-2], hidden[-1]
        # [batch_size, hid_dim*2]
        hidden = torch.cat((hidden_fwd, hidden_bck), dim=1)
        # pred : [batch_size, output_dim]
        return self.fc(self.dropout(hidden))


# TF BiLSTM tfmodel.
def TFLSTM(max_seq_len, vocab_size):
    model = Sequential()
    model.add(Embedding(vocab_size, 64, input_length=max_seq_len))
    model.add(LSTM(64, return_sequences=True))
    model.add(Dropout(0.5))
    model.add(LSTM(64))
    model.add(Dense(256, name='fc1'))
    model.add(Dropout(0.5))
    model.add(Dense(1, name='out'))
    model.add(Activation('sigmoid'))
    return model
