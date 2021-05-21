from bentoml import BentoService, api, artifacts, env
from bentoml.adapters import JsonInput, JsonOutput
from bentoml.frameworks.pytorch import PytorchModelArtifact
from bentoml.service.artifacts.pickle import PickleArtifact
import torch

from train_torch import vocab
from model_torch import device


@env(infer_pip_packages=True, pip_packages=['spacy', 'torchtext'])
@artifacts([PytorchModelArtifact("model"), PickleArtifact("tokenizer")])
class PytorchService(BentoService):
    def predict_sentiment(self, sentence):
        self.artifacts.model.to(device)
        self.artifacts.model.eval()
        tokenized = [tok.text for tok in self.artifacts.tokenizer.tokenizer(sentence)]
        indexed = [vocab.stoi[t] for t in tokenized]
        length = torch.LongTensor([len(indexed)])
        tensor = torch.LongTensor(indexed).to(device)
        tensor = tensor.unsqueeze(1)
        prediction = torch.sigmoid(self.artifacts.model(tensor, length))
        return prediction.item()

    @api(input=JsonInput(), output=JsonOutput())
    def predict(self, parsed_json):
        return self.predict_sentiment(parsed_json['text'])