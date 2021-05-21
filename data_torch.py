import os

import torch
import random
from torchtext.legacy.datasets import IMDB
from torchtext.legacy import data

SEED = 1234


class Dataset:
    def __init__(self, MAX_VOCAB_SIZE, BATCH_SIZE, device):
        self.batch_size = BATCH_SIZE
        self.max_vocab_size = MAX_VOCAB_SIZE
        self.device = device
        self._init_text_label()

    def _init_text_label(self):
        print("\nLoading text and labels")
        TEXT = data.Field(
            tokenize='spacy', tokenizer_language='en_core_web_sm', include_lengths=True
        )
        LABEL = data.LabelField(dtype=torch.float)

        if os.environ.get('IS_IN_DOCKER'):
            root_dir = "$HOME/bundle/PytorchService/dataset"
        else:
            root_dir = "dataset"
        train_data, self.test_data = IMDB.splits(TEXT, LABEL, root=root_dir)

        self.train_data, self.valid_data = train_data.split(random_state=random.seed(SEED))

        TEXT.build_vocab(
            self.train_data,
            max_size=self.max_vocab_size,
            vectors="glove.6B.100d",
            unk_init=torch.Tensor.normal_,
        )

        LABEL.build_vocab(self.train_data)

        self.TEXT, self.LABEL = TEXT, LABEL
        self.vocab = self.TEXT.vocab

    def get_pad_idx(self):
        return self.vocab.stoi[self.TEXT.pad_token]

    def get_unk_idx(self):
        return self.vocab.stoi[self.TEXT.unk_token]

    def get_vocab(self):
        return self.vocab

    def get_iterator(self):
        train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
            (self.train_data, self.valid_data, self.test_data),
            batch_size=self.batch_size,
            sort_within_batch=True,
            device=self.device,
        )
        return train_iterator, valid_iterator, test_iterator