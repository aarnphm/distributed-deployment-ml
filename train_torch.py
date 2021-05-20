import time
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.legacy import data
from torchtext.legacy import datasets

from model_torch import TorchNetwork, device

SEED = 1234

torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

print(f"device: {device}")

print("\nLoading vocab and tokenizer\n")
TEXT = data.Field(tokenize='spacy',
                  tokenizer_language='en_core_web_sm',
                  include_lengths=True)
LABEL = data.LabelField(dtype=torch.float)

train_data, test_data = datasets.IMDB.splits(TEXT, LABEL, root="dataset")

train_data, valid_data = train_data.split(random_state=random.seed(SEED))


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


MAX_VOCAB_SIZE = 25_000
BATCH_SIZE = 64
EMBEDDING_DIM = 100
HIDDEN_DIM = 256
OUTPUT_DIM = 1
N_LAYERS = 2
BIDIRECTIONAL = True
DROPOUT = 0.5
N_EPOCHS = 5

TEXT.build_vocab(train_data,
                 max_size=MAX_VOCAB_SIZE,
                 vectors="glove.6B.100d",
                 unk_init=torch.Tensor.normal_)

LABEL.build_vocab(train_data)

train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
    (train_data, valid_data, test_data),
    batch_size=BATCH_SIZE,
    sort_within_batch=True,
    device=device)

INPUT_DIM = len(TEXT.vocab)
PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]

model = TorchNetwork(INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM, N_LAYERS, BIDIRECTIONAL, DROPOUT, PAD_IDX)

print(f'The model has {count_parameters(model):,} trainable parameters')
print(model)

pretrained_embeddings = TEXT.vocab.vectors

model.embedding.weight.data.copy_(pretrained_embeddings)

UNK_IDX = TEXT.vocab.stoi[TEXT.unk_token]

model.embedding.weight.data[UNK_IDX] = torch.zeros(EMBEDDING_DIM)
model.embedding.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_DIM)
print("\nModel embedding weight: \n")
print(model.embedding.weight.data)

optimizer = optim.Adam(model.parameters())

criterion = nn.BCEWithLogitsLoss()

model = model.to(device)
criterion = criterion.to(device)


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