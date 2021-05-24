import time
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.experimental.vectors import GloVe

from data import Dataset
from config import EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM, N_LAYERS, DROPOUT, BATCH_SIZE, N_EPOCHS, SEED, PAD_TOKEN, \
    UNK_TOKEN
from model import RNN

torch.manual_seed(SEED)
random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_model(pad_idx=None):
    model = RNN(
        INPUT_DIM,
        EMBEDDING_DIM,
        HIDDEN_DIM,
        OUTPUT_DIM,
        N_LAYERS,
        DROPOUT,
        pad_idx
    )

    model = model.to(device)
    return model


def get_pretrained_embedding(init_embedding, pretrained_vectors, vocab, unk_token):
    pretrained_embedding = torch.FloatTensor(init_embedding.weight.clone()).detach()
    pretrained_vocab = pretrained_vectors.get_stoi()

    unk_tokens = []

    # iterate over your vocab's `itos` attribute, a list of tokens within the vocab
    # if the token is in the pre-trained vocab, i.e. if it has a pre-trained vector
    # then replace its row in the pre-trained embedding tensor with the pre-trained vector
    # if the token is NOT in the pre-trained vocab, we leave it initialized to zero
    for idx, token in enumerate(vocab.itos):
        if token in pretrained_vocab:
            pretrained_vector = pretrained_vectors[token]  # FloatTensor pre-trained vector for `token`
            pretrained_embedding[idx] = pretrained_vector  # update the appropriate row in pretrained_embedding
        else:
            unk_tokens.append(token)
    return pretrained_embedding, unk_tokens


def init_params(m):
    if isinstance(m, nn.Embedding):
        nn.init.uniform_(m.weight, -.05, .05)
    elif isinstance(m, nn.LSTM):
        for n, p in m.named_parameters():
            if 'weight_ih' in n:
                i, f, g, o = p.chunk(4)
                nn.init.xavier_uniform_(i)
                nn.init.xavier_uniform_(f)
                nn.init.xavier_uniform_(g)
                nn.init.xavier_uniform_(o)
            elif 'weight_hh' in n:
                i, f, g, o = p.chunk(4)
                nn.init.orthogonal_(i)
                nn.init.orthogonal_(f)
                nn.init.orthogonal_(g)
                nn.init.orthogonal_(o)
            elif 'bias' in n:
                i, f, g, o = p.chunk(4)
                nn.init.zeros_(i)
                nn.init.ones_(f)
                nn.init.zeros_(g)
                nn.init.zeros_(o)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)


def summary(model):
    count_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'\nThe model has {count_params:,} trainable parameters')
    print(f"Model summary:\n{model}\nDetails:\n")
    for n, p in model.named_parameters():
        print(f'name: {n}, shape: {p.shape}')


def calc_acc(predictions, labels):
    top_pred = predictions.argmax(1, keepdim=True)
    correct = top_pred.eq(labels.view_as(top_pred)).sum()
    return correct.float() / labels.shape[0]


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


def epoch_time(start, end):
    elapsed_time = end - start
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


if __name__ == '__main__':

    print("\nLoading IMDB dataset")
    imdb = Dataset()
    vocab = imdb.get_vocab()
    INPUT_DIM = len(vocab)

    model = get_model(pad_idx=imdb.get_pad_idx())

    optimizer = optim.Adam(model.parameters())

    loss_fn = nn.CrossEntropyLoss().to(device)

    train_iterator, test_iterator = imdb.get_iterator()

    glove = GloVe(name="6B", dim=EMBEDDING_DIM)

    pretrained_embedding, unk_tokens = get_pretrained_embedding(model.embedding, glove, vocab, UNK_TOKEN)

    model.embedding.weight.data.copy_(pretrained_embedding)
    summary(model)

    print("\nStart training...")
    best_test_loss = float('inf')

    for epoch in range(N_EPOCHS):

        start_time = time.monotonic()

        train_loss, train_acc = train(model, train_iterator, optimizer, loss_fn)
        test_loss, test_acc = evaluate(model, test_iterator, loss_fn)

        end_time = time.monotonic()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if test_loss < best_test_loss:
            best_test_loss = test_loss
            torch.save(model.state_dict(), '../model/torchnet.pt')

        print(f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')
        print(f'\t Test Loss: {test_loss:.3f} |  Test Acc: {test_acc * 100:.2f}%')