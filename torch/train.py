import time

import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.vocab import GloVe

from data import SEED, Dataset
from config import EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM, N_LAYERS, DROPOUT, BATCH_SIZE, N_EPOCHS
from model import RNN


torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_model():
    model = RNN(
        INPUT_DIM,
        EMBEDDING_DIM,
        HIDDEN_DIM,
        OUTPUT_DIM,
        N_LAYERS,
        DROPOUT,
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


def epoch_time(start, end):
    elapsed_time = end - start
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


if __name__ == '__main__':

    print("\nLoading IMDB dataset")
    imdb = Dataset(BATCH_SIZE)
    vocab = imdb.get_vocab()
    INPUT_DIM = len(vocab)

    model = get_model()
    print(f'\nThe model has {count_parameters(model):,} trainable parameters')
    print(model)

    optimizer = optim.Adam(model.parameters())

    loss_fn = nn.BCEWithLogitsLoss().to(device)

    train_iterator, valid_iterator, test_iterator = imdb.get_iterator()

    glove = GloVe(name="6B", dim=EMBEDDING_DIM)

    # create a tensor used for holding the pre-trained vectors for each element of the vocab
    pretrained_embedding = torch.zeros(INPUT_DIM, EMBEDDING_DIM)

    # get the pretrained vector's vocab, Dict[str, int]
    pretrained_vocab = glove.vectors.get_stoi()

    # iterate over your vocab's `itos` attribute, a list of tokens within the vocab
    # if the token is in the pre-trained vocab, i.e. if it has a pre-trained vector
    # then replace its row in the pre-trained embedding tensor with the pre-trained vector
    # if the token is NOT in the pre-trained vocab, we leave it initialized to zero
    for idx, token in enumerate(vocab.itos):
        if token in pretrained_vocab:
            pretrained_vector = glove[token]  # pretrained_vector is a FloatTensor pre-trained vector for `token`
            pretrained_embedding[idx] = pretrained_vector  # update the appropriate row in pretrained_embedding

    model.embedding.weight.data.copy_(pretrained_embedding)


    print("\nStart training\n")
    best_valid_loss = float('inf')

    for epoch in range(N_EPOCHS):

        start_time = time.time()

        train_loss, train_acc = train(model, train_iterator, optimizer, loss_fn)
        valid_loss, valid_acc = evaluate(model, valid_iterator, loss_fn)

        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), '../model/torchnet.pt')

        print(f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc * 100:.2f}%')