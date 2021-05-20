from bentoml import BentoService, api, artifacts, env
from bentoml.adapters import JsonInput
from bentoml.frameworks.keras import KerasModelArtifact
from bentoml.frameworks.pytorch import PytorchModelArtifact
from bentoml.service.artifacts.common import PickleArtifact
from tensorflow.keras.preprocessing import sequence, text
import torch
import torch.nn.functional as F
from data_tf import preprocess
from train_torch import TEXT
from model import device


@env(infer_pip_packages=True)
@artifacts([KerasModelArtifact('model'), PickleArtifact('tokenizer')])
class TensorflowService(BentoService):
    def word_to_index(self, word):
        if word in self.artifacts.tokenizer and self.artifacts.tokenizer[word] <= 5000:
            return self.artifacts.tokenizer[word]
        else:
            return self.artifacts.tokenizer["<OOV>"]

    def preprocessing(self, text_str):
        proc = text.text_to_word_sequence(preprocess(text_str))
        tokens = list(map(self.word_to_index, proc))
        return tokens

    @api(input=JsonInput())
    def predict(self, parsed_json):
        # single pred
        raw = self.preprocessing(parsed_json['text'])
        print(raw)
        input_data = [raw[: n + 1] for n in range(len(raw))]
        input_data = sequence.pad_sequences(input_data, maxlen=100, padding="post")
        return self.artifacts.model.predict(input_data, verbose=1)


@env(infer_pip_packages=True)
@artifacts([PytorchModelArtifact("cnn")])
class PytorchService(BentoService):
    @api(input=JsonInput())
    def predict(self, parsed_json, min_len=5):
        src_text = parsed_json.get("text")
        model = self.artifacts.cnn.get("model")
        nlp = self.artifacts.cnn.get("tokenizer")
        model.eval()
        tokenized = [tok.text for tok in nlp.tokenizer(src_text)]
        if len(tokenized) < min_len:
            tokenized += ['<pad>'] * (min_len - len(tokenized))
        indexed = [TEXT.vocab.stoi[t] for t in tokenized]
        tensor = torch.LongTensor(indexed).to(device)
        tensor = tensor.unsqueeze(0)
        prediction = F.sigmoid(model(tensor))
        return prediction.item()
