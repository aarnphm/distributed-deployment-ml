import torch
import torch.nn as nn


class RNN(nn.Module):
    def __init__(
            self,
            input_dim,
            embedding_dim,
            hidden_dim,
            output_dim,
            n_layers,
            dropout,
            pad_idx,
    ):
        super().__init__()

        self.embedding = nn.Embedding(input_dim, embedding_dim, padding_idx=pad_idx)

        # bidirectional is set to True by default
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=n_layers,
            bidirectional=True,
            dropout=dropout,
        )

        self.fc = nn.Linear(hidden_dim * 2, output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, text, length):
        # text = [seq_len, batch]
        # length = [batch]
        embedded = self.dropout(self.embedding(text))

        # embedded = [seq_len, batch, emb_dim]

        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, length, enforce_sorted=False)

        packed_output, (hidden, cell) = self.lstm(packed_embedded)
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output)

        # output = [sent len, batch size, hid dim * num directions]
        # output over padding tokens are zero tensors

        # hidden = [num layers * num directions, batch size, hid dim]
        # cell = [num layers * num directions, batch size, hid dim]

        # concat the final forward (hidden[-2,:,:]) and backward (hidden[-1,:,:]) hidden layers
        # and apply dropout

        hidden = self.dropout(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))

        # hidden = [batch size, hid dim * num directions]

        return self.fc(hidden)