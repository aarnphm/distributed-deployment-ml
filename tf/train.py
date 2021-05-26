import json

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import RMSprop

from data import IMDB
from model import RNN

# CONSTANT
VOCAB_SIZE = 5000
MAX_SEQ_LEN = 100
SEED = 1234

gpu = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpu[0], True)  # gpu name: /GPU:0

model = RNN(MAX_SEQ_LEN, VOCAB_SIZE)
model.summary()
model.compile(loss='binary_crossentropy', optimizer=RMSprop(), metrics=['accuracy'])

imdb = IMDB(MAX_SEQ_LEN, VOCAB_SIZE)

if __name__ == '__main__':
    # people can find their devices using tf.config.list_physical_devices("GPU") # "CPU"
    with tf.device("/GPU:0"):
        # Model Training
        model.fit(
            imdb.X_train,
            imdb.Y_train,
            batch_size=512,
            epochs=10,
            validation_split=0.2,
            callbacks=[EarlyStopping(patience=2, verbose=1)],
        )

        # Run model on test set
        accr = model.evaluate(imdb.X_test, imdb.Y_test)
        print(
            'Test set\n  Loss: {:0.4f}\n  Accuracy: {:0.2f}'.format(
                accr[0], accr[1] * 100
            )
        )

        # save weights as HDF5
        model.save("../model/tf/weights.h5")
        print("Saved model to disk")

        # save model as JSON
        model_json = model.to_json()
        with open("../model/tf/model.json", "w") as file:
            file.write(model_json)

        # save tokenizer as JSON
        tokenizer_json = imdb.tokenizer.to_json()
        with open("../model/tf/tokenizer.json", 'w', encoding='utf-8') as file:
            file.write(json.dumps(tokenizer_json, ensure_ascii=True))