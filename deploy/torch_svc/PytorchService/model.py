import torch
import torch.nn as nn


def summary(model):
    count_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'\nThe model has {count_params:,} trainable parameters')
    print(f"Model summary:\n{model}\nDetails:")
    for n, p in model.named_parameters():
        print(f'name: {n}, shape: {p.shape}')


class TextClassificationModel(nn.Module):

    def __init__(self, vocab_size, embed_dim, num_class):
        super().__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=True)
        self.fc = nn.Linear(embed_dim, num_class)
        self.init_weights()

    def init_weights(self):
        init_range = 0.5
        self.embedding.weight.data.uniform_(-init_range, init_range)
        self.fc.weight.data.uniform_(-init_range, init_range)
        self.fc.bias.data.zero_()

    def forward(self, text, offsets=None):
        embedded = self.embedding(text, offsets=offsets)
        return self.fc(embedded)


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