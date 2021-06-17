import os
import json
from tensorflow import config

from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing.text import tokenizer_from_json

from bento_service import TensorflowService

gpu = config.experimental.list_physical_devices('GPU')
print(gpu)
config.experimental.set_memory_growth(gpu[0], True)


def load_tokenizer():
    with open('../model/tf/tokenizer.json', 'r') as f:
        data = json.load(f)
        tokenizer = tokenizer_from_json(data)
        j = tokenizer.get_config()['word_index']
        return json.loads(j)


def load_model():
    # load json and create model
    json_file = open('../model/tf/model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.load_weights("../model/tf/weights.h5")
    return model


model = load_model()
tokenizer = load_tokenizer()

bento_svc = TensorflowService()
bento_svc.pack('model', model)
bento_svc.pack('tokenizer', tokenizer)

saved_path = bento_svc.save()

if __name__ == '__main__':
    print("\nExample run:")
    print(bento_svc.predict({'text': "I love you"}))
    print("---")

    print(bento_svc.predict({'text': "I hate you"}))
    print("---")
    print(f"saved model path: {saved_path}")
