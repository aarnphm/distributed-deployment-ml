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
