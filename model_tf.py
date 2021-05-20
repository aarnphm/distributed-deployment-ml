from tensorflow.keras.layers import Activation, Dense, Dropout, Embedding, LSTM
from tensorflow.keras.models import Sequential


# TF BiLSTM tfmodel.
def TFNetwork(max_seq_len, vocab_size):
    model = Sequential()
    model.add(Embedding(vocab_size, 64, input_length=max_seq_len))
    model.add(LSTM(64, return_sequences=True))
    model.add(Dropout(0.5))
    model.add(LSTM(64))
    model.add(Dense(256, name='fc1'))
    model.add(Dropout(0.5))
    model.add(Dense(1, name='out'))
    model.add(Activation('sigmoid'))
    return model
