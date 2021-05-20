import torch
import random
from torchtext.legacy import datasets
from torchtext.legacy import data

SEED = 1234


class Dataset:
    def __init__(self, MAX_VOCAB_SIZE, BATCH_SIZE):
        TEXT = data.Field(
            tokenize='spacy', tokenizer_language='en_core_web_sm', include_lengths=True
        )
        LABEL = data.LabelField(dtype=torch.float)

        train_data, test_data = datasets.IMDB.splits(TEXT, LABEL, root="dataset")

        train_data, valid_data = train_data.split(random_state=random.seed(SEED))

        TEXT.build_vocab(
            train_data,
            max_size=MAX_VOCAB_SIZE,
            vectors="glove.6B.100d",
            unk_init=torch.Tensor.normal_,
        )

        LABEL.build_vocab(train_data)

        train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
            (train_data, valid_data, test_data),
            batch_size=BATCH_SIZE,
            sort_within_batch=True,
            device=device,
        )
