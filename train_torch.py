import time

import torch
import torch.nn as nn
import torch.optim as optim

from data_torch import SEED, Dataset
from model_torch import BiLSTM, device

torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

BIDIRECTIONAL = True
MAX_VOCAB_SIZE = 25000
BATCH_SIZE = 64
EMBEDDING_DIM = 100
HIDDEN_DIM = 256
OUTPUT_DIM = 1
N_LAYERS = 2
DROPOUT = 0.5
N_EPOCHS = 5

def get_model():

    model = BiLSTM(
        INPUT_DIM,
        EMBEDDING_DIM,
        HIDDEN_DIM,
        OUTPUT_DIM,
        N_LAYERS,
        BIDIRECTIONAL,
        DROPOUT,
        PAD_IDX,
    )

    model = model.to(device)
    return model



def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def binary_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """

    # round predictions to the closest integer
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float()  # convert into float for division
    acc = correct.sum() / len(correct)
    return acc


def train(model, iterator, optimizer, criterion):
    ep_loss, ep_acc = 0, 0

    model.train()

    for batch in iterator:
        labels = batch.label
        text, lengths = batch.text
        labels, text = labels.to(device), text.to(device)

        optimizer.zero_grad()

        predictions = model(text, lengths).squeeze(1)

        loss = criterion(predictions, labels)

        acc = binary_accuracy(predictions, labels)

        loss.backward()
        optimizer.step()

        ep_loss += loss.item()
        ep_acc += acc.item()

    return ep_loss / len(iterator), ep_acc / len(iterator)


def evaluate(model, iterator, criterion):
    ep_loss, ep_acc = 0, 0

    model.eval()

    with torch.no_grad():
        for batch in iterator:
            labels = batch.label
            text, lengths = batch.text
            labels, text = labels.to(device), text.to(device)

            predictions = model(text, lengths).squeeze(1)

            loss = criterion(predictions, labels)

            acc = binary_accuracy(predictions, labels)

            ep_loss += loss.item()
            ep_acc += acc.item()

    return ep_loss / len(iterator), ep_acc / len(iterator)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


if __name__ == '__main__':

    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

    BIDIRECTIONAL = True
    MAX_VOCAB_SIZE = 25000
    BATCH_SIZE = 64
    EMBEDDING_DIM = 100
    HIDDEN_DIM = 256
    OUTPUT_DIM = 1
    N_LAYERS = 2
    DROPOUT = 0.5
    N_EPOCHS = 5

    imdb = Dataset(MAX_VOCAB_SIZE, BATCH_SIZE, device)
    vocab = imdb.get_vocab()

    INPUT_DIM = len(vocab)
    PAD_IDX = imdb.get_pad_idx()


    model = BiLSTM(
        INPUT_DIM,
        EMBEDDING_DIM,
        HIDDEN_DIM,
        OUTPUT_DIM,
        N_LAYERS,
        BIDIRECTIONAL,
        DROPOUT,
        PAD_IDX,
    )

    print("hello world")
    model = model.to(device)
    print(f'The model has {count_parameters(model):,} trainable parameters')
    print(model)

    UNK_IDX = imdb.get_unk_idx()

    train_iterator, valid_iterator, test_iterator = imdb.get_iterator()
    pretrained_embeddings = vocab.vectors

    model.embedding.weight.data.copy_(pretrained_embeddings)

    model.embedding.weight.data[UNK_IDX] = torch.zeros(EMBEDDING_DIM)
    model.embedding.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_DIM)

    optimizer = optim.Adam(model.parameters())

    criterion = nn.BCEWithLogitsLoss()

    criterion = criterion.to(device)

    print("\nStart training\n")
    best_valid_loss = float('inf')

    for epoch in range(N_EPOCHS):

        start_time = time.time()

        train_loss, train_acc = train(model, train_iterator, optimizer, criterion)
        valid_loss, valid_acc = evaluate(model, valid_iterator, criterion)

        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'model/torchnet.pt')

        print(f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc * 100:.2f}%')