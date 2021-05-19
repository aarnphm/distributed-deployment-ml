from model import TFLSTM
from data import TfDataset
import json
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import EarlyStopping

import os
import random
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torchtext
import torchtext.experimental
import torchtext.experimental.vectors

from data import (
    TorchDataset,
    calc_acc,
    ep_time,
    get_config,
    get_pretrained_embedding,
    init_params,
)
from model import TorchLSTM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config = get_config(
    os.path.join(os.path.abspath(os.path.dirname(__file__)), "config.yml")
)

print("\nSetting up embedding.\n")
torch.manual_seed(config["seed"])
random.seed(config["seed"])
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# CONSTANT
MODEL_PATH = os.path.join(os.getcwd(), "models", config["model_name"] + ".pt")
VOCAB_SIZE = 5000
MAX_SEQ_LEN = 100

# setup for torch.
dataloader = TorchDataset(
    config["max_len"], config["max_size"], config["batch_size"], config["pad_token"]
)
train_iterator, test_iterator, valid_iterator = dataloader.get_iterator()
print("Loaded iterator, generating vocab...")
vocab = dataloader.get_vocab()
tokenizer = dataloader.get_tokenizer()

pad_idx = vocab[config["pad_token"]]
input_dim = len(vocab)

imdb = TfDataset(MAX_SEQ_LEN, VOCAB_SIZE)


def tf_train(tfmodel):

    # Model Training
    tfmodel.fit(
        imdb.X_train,
        imdb.Y_train,
        batch_size=512,
        epochs=4,
        validation_split=0.2,
        callbacks=[EarlyStopping(patience=2, verbose=1)],
    )

    # Run model on test set
    accr = tfmodel.evaluate(imdb.X_test, imdb.Y_test)
    print(
        'Test set\n  Loss: {:0.4f}\n  Accuracy: {:0.2f}'.format(accr[0], accr[1] * 100)
    )

    # save weights as HDF5
    tfmodel.save("model/tfweights.h5")
    print("Saved model to disk")

    # save model as JSON
    model_json = tfmodel.to_json()
    with open("model/tfmodel.json", "w") as file:
        file.write(model_json)

    # save tokenizer as JSON
    tokenizer_json = imdb.tokenizer.to_json()
    with open("model/tftokenizer.json", 'w', encoding='utf-8') as file:
        file.write(json.dumps(tokenizer_json, ensure_ascii=True))


class TorchModel:
    def __init__(self):
        model = TorchLSTM(
            input_dim,
            config["emb_dim"],
            config["hid_dim"],
            config["output_dim"],
            config["n_layers"],
            config["dropout"],
            pad_idx,
        )

        glove = torchtext.experimental.vectors.GloVe(name="6B", dim=config["emb_dim"])
        model.apply(init_params)

        pretrained_embedding, _ = get_pretrained_embedding(
            model.embedding, glove, vocab, config["unk_token"]
        )

        model.embedding.weight.data.copy_(pretrained_embedding)
        model.embedding.weight.data[pad_idx] = torch.zeros(config["emb_dim"])

        self.optimizer = optim.Adam(model.parameters())

        criterion = nn.CrossEntropyLoss()

        self.model = model.to(device)
        self.criterion = criterion.to(device)

    @staticmethod
    def train(model, iterator, optimizer, criterion):
        ep_loss, ep_acc = 0, 0

        model.train()

        for labels, text, lengths in iterator:
            labels, text = labels.to(device), text.to(device)

            optimizer.zero_grad()

            predictions = model(text, lengths)

            loss = criterion(predictions, labels)

            acc = calc_acc(predictions, labels)

            loss.backward()
            optimizer.step()

            ep_loss += loss.item()
            ep_acc += acc.item()

        return ep_loss / len(iterator), ep_acc / len(iterator)

    @staticmethod
    def evaluate(model, iterator, criterion):
        ep_loss, ep_acc = 0, 0

        model.eval()

        with torch.no_grad():
            for labels, text, lengths in iterator:
                labels, text = labels.to(device), text.to(device)

                predictions = model(text, lengths)

                loss = criterion(predictions, labels)

                acc = calc_acc(predictions, labels)

                ep_loss += loss.item()
                ep_acc += acc.item()

        return ep_loss / len(iterator), ep_acc / len(iterator)

    def train_loop(self, train_iter, valid_iter):
        best_valid_loss = float("inf")
        print("Start training...")
        for epoch in range(config["n_epochs"]):

            start_time = time.monotonic()

            train_loss, train_acc = self.train(
                self.model, train_iter, self.optimizer, self.criterion
            )
            valid_loss, valid_acc = self.evaluate(
                self.model, valid_iter, self.criterion
            )

            end_time = time.monotonic()

            epoch_mins, epoch_secs = ep_time(start_time, end_time)

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                torch.save(self.model.state_dict(), MODEL_PATH)

            print(f"Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s")
            print(f"\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%")
            print(f"\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%")


if __name__ == "__main__":
    import argparse

    print("\nSetting up for TorchModel.\n")
    m = TorchModel()
    print("\nSetting up for TfModel.\n")
    tfmodel = TFLSTM(MAX_SEQ_LEN, VOCAB_SIZE)
    tfmodel.summary()
    tfmodel.compile(
        loss='binary_crossentropy', optimizer=RMSprop(), metrics=['accuracy']
    )

    parser = argparse.ArgumentParser(description="training model")
    parser.add_argument(
        "--train",
        action="store_true",
        default=False,
        help="whether or not to train a new pytorch model",
    )
    parser.add_argument(
        "--tf",
        action="store_true",
        default=False,
        help="whether or not to train a new tensorflow model",
    )
    args = parser.parse_args()
    if args.train:
        m.train_loop(train_iterator, valid_iterator)
    if args.tf:
        tf_train(tfmodel)
