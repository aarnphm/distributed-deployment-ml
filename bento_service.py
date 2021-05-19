from bentoml import api, env, BentoService, artifacts
from bentoml.artifact import KerasModelArtifact, PickleArtifact
from bentoml.handlers import JsonHandler
from tensorflow.keras.preprocessing import sequence, text
from data import preprocess

import torch
import torch.nn as nn
from bentoml import BentoService, api, artifacts, env
from bentoml.adapters import JsonInput, JsonOutput
from bentoml.service.artifacts.common import PickleArtifact
from bentoml.frameworks.pytorch import PytorchModelArtifact
from train import vocab, device


@env(auto_pip_dependencies=True, infer_pip_packages=True)
@artifacts([PytorchModelArtifact("torchmodel"), PickleArtifact("torchtokenizer")])
class TorchService(BentoService):
    def model_pred(self, sentence):
        self.artifacts.model.eval()

        tokens = self.artifacts.torchtokenizer.tokenize(sentence)
        length = torch.LongTensor([len(tokens)]).to(device)
        idx = [vocab.stoi[token] for token in tokens]
        tensor = torch.LongTensor(idx).unsqueeze(-1).to(device)

        prediction = self.artifacts.torchmodel(tensor, length)
        probabilities = nn.softmax(prediction, dim=-1)
        return probabilities.squeeze()[-1].item()

    @api(input=JsonInput(), output=JsonOutput())
    def predict(self, parsed_json):
        return self.model_pred(parsed_json["text"])


@env(infer_pip_packages=True)
@artifacts([KerasModelArtifact('kerasmodel'), PickleArtifact('kerastokenizer')])
class TensorflowService(BentoService):
    def word_to_index(self, word):
        if (
            word in self.artifacts.kerastokenizer
            and self.artifacts.kerastokenizer[word] <= 5000
        ):
            return self.artifacts.kerastokenizer[word]
        else:
            return self.artifacts.kerastokenizer["<OOV>"]

    def preprocessing(self, text_str):
        proc = text.text_to_word_sequence(preprocess(text_str))
        tokens = list(map(self.word_to_index, proc))
        return tokens

    @api(JsonHandler)
    def predict(self, parsed_json):
        # single pred
        raw = self.preprocessing(parsed_json['text'])
        input_data = [raw[: n + 1] for n in range(len(raw))]
        input_data = sequence.pad_sequences(input_data, maxlen=100, padding="post")
        return self.artifacts.kerasmodel.predict(input_data, verbose=1)
