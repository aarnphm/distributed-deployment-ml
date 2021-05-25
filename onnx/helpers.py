import time

import torch
from torchtext.datasets import AG_NEWS
from torchtext.data.utils import get_tokenizer
from collections import Counter
from torchtext.vocab import Vocab

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
EPOCHS = 10  # epoch
LR = 5  # learning rate
BATCH_SIZE = 64  # batch size for training
EMSIZE = 64

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

def get_tokenizer_vocab(tokenizer_fn='basic_english', root_data_dir='../dataset'):
    print('Getting tokenizer and vocab...')
    tokenizer = get_tokenizer(tokenizer_fn)
    train_ = AG_NEWS(root=root_data_dir, split='train')
    counter = Counter()
    for (label, line) in train_:
        counter.update(tokenizer(line))
    vocab = Vocab(counter, min_freq=1)
    return tokenizer, vocab


def get_pipeline(tokenizer, vocab):
    print('Setup pipeline...')
    text_pipeline = lambda x: [vocab[token] for token in tokenizer(x)]
    label_pipeline = lambda x: int(x) - 1
    return text_pipeline, label_pipeline


def get_model_params(vocab):
    print('Setup model params...')
    train_iter = AG_NEWS(root='../dataset', split='train')
    num_class = len(set([label for (label, text) in train_iter]))
    vocab_size = len(vocab)
    return vocab_size, EMSIZE, num_class