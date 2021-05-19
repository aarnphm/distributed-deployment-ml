from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.preprocessing.text import Tokenizer
import pandas as pd

import string
import collections
import os
import random
import torch
import torch.nn as nn
import torchtext
import torchtext.experimental
import torchtext.experimental.vectors
import yaml

from torchtext.experimental.datasets.text_classification import (
    TextClassificationDataset,
)


def get_config(fpath: str) -> dict:
    with open(fpath, "r") as f:
        parsed = yaml.load(f, Loader=yaml.FullLoader)
    f.close()
    return parsed


def get_pretrained_embedding(init_embed, pretrained_vectors, vocab, unk_token):
    pretrained_embedding = torch.FloatTensor(init_embed.weight.clone()).detach()
    pretrained_vocab = pretrained_vectors.vectors.get_stoi()

    unk_tokens = []

    for idx, token in enumerate(vocab.itos):
        if token in pretrained_vocab:
            pretrained_vector = pretrained_vectors[token]
            pretrained_embedding[idx] = pretrained_vector
        else:
            unk_tokens.append(token)

    return pretrained_embedding, unk_tokens


def init_params(m: nn.Module):
    if isinstance(m, nn.Embedding):
        nn.init.uniform_(m.weight, -0.05, 0.05)
    elif isinstance(m, nn.LSTM):
        for n, p in m.named_parameters():
            if "weight_ih" in n:
                i, f, g, o = p.chunk(4)
                nn.init.xavier_uniform_(i)
                nn.init.xavier_uniform_(f)
                nn.init.xavier_uniform_(g)
                nn.init.xavier_uniform_(o)
            elif "weight_hh" in n:
                i, f, g, o = p.chunk(4)
                nn.init.orthogonal_(i)
                nn.init.orthogonal_(f)
                nn.init.orthogonal_(g)
                nn.init.orthogonal_(o)
            elif "bias" in n:
                i, f, g, o = p.chunk(4)
                nn.init.zeros_(i)
                nn.init.ones_(f)
                nn.init.zeros_(g)
                nn.init.zeros_(o)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)


def sequential_transforms(*transforms):
    def func(inputs):
        for transform in transforms:
            inputs = transform(inputs)
        return inputs

    return func


def to_tensor(dtype):
    def func(ids_list):
        return torch.tensor(ids_list).to(dtype)

    return func


def vocab_func(vocab):
    def func(tok_iter):
        return [vocab[tok] for tok in tok_iter]

    return func


def ep_time(start_time, end_time):
    elapsed = end_time - start_time
    elapsed_mins = int(elapsed / 60)
    elapsed_secs = int(elapsed - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def calc_acc(predictions, labels):
    top_pred = predictions.argmax(1, keepdim=True)
    correct = top_pred.eq(labels.view_as(top_pred)).sum()
    acc = correct.float() / labels.shape[0]
    return acc


def get_train_valid_split(raw_train, split_ratio=0.7):
    raw_train = list(raw_train)
    random.shuffle(raw_train)

    n_train_ex = int(len(raw_train) * split_ratio)
    train_data = raw_train[:n_train_ex]
    valid_data = raw_train[n_train_ex:]
    return train_data, valid_data


def gen_vocab(raw_data, tokenizer, **vocab_kwargs):
    token_freqs = collections.Counter()

    for label, text in raw_data:
        tokens = tokenizer.tokenize(text)
        token_freqs.update(tokens)

    vocab = torchtext.vocab.Vocab(token_freqs, **vocab_kwargs)

    return vocab


def process_raw(raw_data, tokenizer, vocab):
    raw_data = [(label, text) for (label, text) in raw_data]
    text_trans = sequential_transforms(
        tokenizer.tokenize, vocab_func(vocab), to_tensor(dtype=torch.long)
    )
    label_trans = sequential_transforms(to_tensor(dtype=torch.long))

    transforms = (label_trans, text_trans)

    return TextClassificationDataset(raw_data, vocab, transforms)


class TorchTokenizer:
    def __init__(self, fn="basic_english", lower=True, max_len=None):
        self.tokenize_fn = torchtext.data.utils.get_tokenizer(fn)
        self.lower = lower
        self.max_len = max_len

    def tokenize(self, s):
        tokens = self.tokenize_fn(s)

        if self.lower:
            tokens = [token.lower() for token in tokens]

        if self.max_len is not None:
            tokens = tokens[: self.max_len]

        return tokens


class Collator:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def collate(self, batch):
        labels, text = zip(*batch)
        labels, lengths = (
            torch.LongTensor(labels),
            torch.LongTensor([len(x) for x in text]),
        )

        text = nn.utils.rnn.pad_sequence(text, padding_value=self.pad_idx)
        return labels, text, lengths


class TorchDataset:
    def __init__(self, max_len, max_size, batch_size, pad_token):
        self.MAX_LEN = max_len
        self.MAX_SIZE = max_size
        self.BATCH_SIZE = batch_size

        raw_train, raw_test = torchtext.datasets.IMDB(
            root=os.path.join(os.path.abspath(os.path.dirname(__file__)), '.data')
        )
        raw_train, raw_valid = get_train_valid_split(raw_train)

        self.tokenizer = TorchTokenizer(max_len=max_len)
        self.vocab = gen_vocab(raw_train, self.tokenizer, max_size=max_size)
        self.collator = Collator(self.vocab[pad_token])

        self.train_data = process_raw(raw_train, self.tokenizer, self.vocab)
        self.test_data = process_raw(raw_test, self.tokenizer, self.vocab)
        self.valid_data = process_raw(raw_valid, self.tokenizer, self.vocab)

    def get_vocab(self):
        return self.vocab

    def get_tokenizer(self):
        return self.tokenizer

    def get_iterator(self):
        train_iterator = torch.utils.data.DataLoader(
            self.train_data,
            self.BATCH_SIZE,
            shuffle=True,
            collate_fn=self.collator.collate,
        )
        valid_iterator = torch.utils.data.DataLoader(
            self.valid_data,
            self.BATCH_SIZE,
            shuffle=False,
            collate_fn=self.collator.collate,
        )
        test_iterator = torch.utils.data.DataLoader(
            self.test_data,
            self.BATCH_SIZE,
            shuffle=False,
            collate_fn=self.collator.collate,
        )
        return train_iterator, test_iterator, valid_iterator


def preprocess(s):
    return strip_punctuation(remove_br(s.lower()))


def strip_punctuation(s):
    for c in string.punctuation + "’":
        s = s.replace(c, "")
    return s


def remove_br(s):
    return s.replace("<br /><br />", "")


class TfDataset:
    def __init__(self, max_seq_len, vocab_size, dataset="imdb"):
        self.MAX_SEQ_LEN = max_seq_len
        self.VOCAB_SIZE = vocab_size

        if dataset == "imdb":
            print('Loading IMDB dataset')
            df = pd.read_csv('.data/imdb.csv', names=["X", "Y"], skiprows=1)

            # cast X to str and preprocess
            df['X'] = df.X.apply(str)
            df['X'] = df.X.apply(preprocess)

            X = df.X
            Y = df.Y

        # encode labels
        label_encoder = LabelEncoder()
        Y = label_encoder.fit_transform(Y)
        Y = Y.reshape(-1, 1)

        # 15/85 train test split
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(
            X, Y, test_size=0.15
        )

        self.tokenizer = Tokenizer(num_words=self.VOCAB_SIZE, oov_token="<OOV>")
        self.tokenizer.fit_on_texts(self.X_train)

        self.tokenize()
        self.pad()

        print(self.X_train[:30])
        print(self.Y_train[:30])

    def tokenize(self):
        self.X_train = self.tokenizer.texts_to_sequences(self.X_train)
        self.X_test = self.tokenizer.texts_to_sequences(self.X_test)

    def pad(self):
        self.X_train = sequence.pad_sequences(
            self.X_train, maxlen=self.MAX_SEQ_LEN, padding="post"
        )
        self.X_test = sequence.pad_sequences(
            self.X_test, maxlen=self.MAX_SEQ_LEN, padding="post"
        )
