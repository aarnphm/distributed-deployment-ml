import os
from collections import Counter

import torch
import random

from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import random_split, DataLoader
from torchtext.datasets import IMDB
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import Vocab

SEED = 1234


class Dataset:
    def __init__(self, batch_size):
        self.BATCH_SIZE = batch_size
        self.tokenizer = get_tokenizer("spacy", "en_core_web_sm")
        self.train_data, self.test_data = IMDB(root="../dataset")
        self.train_data, self.valid_data = random_split(self.train_data, [20000, 5000])
        self.vocab = self._init_vocab()
        self.text_transform = lambda x: [self.vocab['<BOS>']] + [self.vocab[token] for token in self.tokenizer(x)] + [
            self.vocab['<EOS>']]
        self.label_transform = lambda x: 1 if x == 'pos' else 0

    def get_vocab(self):
        return self.vocab

    def get_tokenizer(self):
        return self.tokenizer

    def _init_vocab(self):
        counter = Counter()
        for label, text in self.train_data:
            counter.update(self.tokenizer(text))
        return Vocab(counter, min_freq=10, specials=('<unk>', '<BOS>', '<EOS>', '<PAD>'))

    @staticmethod
    def collate_batch(self, batch):
        label_list, text_list = [], []
        for (_label, _text) in batch:
            label_list.append(self.label_transform(_label))
            processed_text = torch.tensor(self.text_transform(_text))
            text_list.append(processed_text)
        return torch.tensor(label_list), pad_sequence(text_list, padding_value=3.0)

    def get_iterator(self):
        train_iter = DataLoader(list(self.train_data), batch_size=self.BATCH_SIZE, shuffle=True,
                                collate_fn=self.collate_batch)
        valid_iter = DataLoader(list(self.valid_data), batch_size=self.BATCH_SIZE, shuffle=True,
                                collate_fn=self.collate_batch)
        test_iter = DataLoader(list(self.test_data), batch_size=self.BATCH_SIZE, shuffle=True,
                               collate_fn=self.collate_batch)
        return train_iter, valid_iter, test_iter