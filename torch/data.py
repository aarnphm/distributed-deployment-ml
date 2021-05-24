from collections import Counter

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torchtext.data.utils import get_tokenizer
from torchtext.experimental.datasets import IMDB
from torchtext.experimental.datasets.text_classification import TextClassificationDataset
from torchtext.experimental.functional import sequential_transforms, vocab_func, totensor
from torchtext.vocab import Vocab

from config import MAX_LENGTH, MAX_VOCAB_SIZE, PAD_TOKEN, BATCH_SIZE


def build_vocab(raw_data, tokenizer, **vocab_kwargs):
    token_freq = Counter()
    for label, text in raw_data:
        tokens = tokenizer.tokenize(text)
        token_freq.update(tokens)
    return Vocab(token_freq, **vocab_kwargs)


def process_raw_data(raw_data, tokenizer, vocab):
    raw = [(label, text) for label, text in raw_data]
    text_transform = sequential_transforms(tokenizer.tokenize, vocab_func(vocab), totensor(dtype=torch.long))
    label_transform = sequential_transforms(lambda x: 1 if x == "pos" else 0, totensor(dtype=torch.long))
    transforms = (label_transform, text_transform)

    dataset = TextClassificationDataset(raw, vocab, transforms)
    return dataset


class Tokenizer:
    def __init__(self, tokenizer_fn='spacy', language='en', lower=True, max_length=None):
        self.tokenize_fn = get_tokenizer(tokenizer_fn, language)
        self.lower = lower
        self.max_length = max_length

    def tokenize(self, s):
        tokens = self.tokenize_fn(s)
        if self.lower:
            tokens = [token.lower() for token in tokens]
        if self.max_length is not None:
            tokens = tokens[:self.max_length]
        return tokens


class Collator:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def collate(self, batch):
        labels, text = zip(*batch)
        labels = torch.LongTensor(labels)
        text_len = torch.LongTensor([len(x) for x in text])
        text = pad_sequence(text, padding_value=self.pad_idx)
        return labels, text, text_len


class Dataset:
    def __init__(self):
        raw_train, raw_test = IMDB(root="../dataset", split=("train", "test"))
        raw_train, raw_test = list(raw_train), list(raw_test)
        self.tokenizer = Tokenizer(max_length=MAX_LENGTH)
        self.vocab = build_vocab(raw_train, tokenizer=self.tokenizer, max_size=MAX_VOCAB_SIZE)
        self.train_data = process_raw_data(raw_train, self.tokenizer, self.vocab)
        self.test_data = process_raw_data(raw_test, self.tokenizer, self.vocab)
        pad_idx = self.vocab[PAD_TOKEN]
        self.collator = Collator(pad_idx=pad_idx)

    def get_tokenizer(self):
        return self.tokenizer

    def get_pad_idx(self):
        return self.vocab[PAD_TOKEN]

    def get_vocab(self):
        return self.vocab

    def get_iterator(self):
        train_iter = DataLoader(self.train_data, BATCH_SIZE, shuffle=True, collate_fn=self.collator.collate)
        test_iter = DataLoader(self.test_data, BATCH_SIZE, shuffle=False, collate_fn=self.collator.collate)
        return train_iter, test_iter