from bentoml import BentoService, api, artifacts, env
from bentoml.adapters import JsonInput
from bentoml.frameworks.pytorch import PytorchModelArtifact
from bentoml.service.artifacts.pickle import PickleArtifact
import torch
import torch.nn.functional as F

from train_torch import TEXT
from model_torch import device


@env(infer_pip_packages=True)
@artifacts([PytorchModelArtifact("model"), PickleArtifact("tokenizer")])
class PytorchService(BentoService):
    @api(input=JsonInput())
    def predict(self, parsed_json):
        sentence = parsed_json.get("text")
        model = self.artifacts.model
        nlp = self.artifacts.tokenizer
        model.eval()
        tokenized = [tok.text for tok in nlp.tokenizer(sentence)]
        indexed = [TEXT.vocab.stoi[t] for t in tokenized]
        length_tensor = torch.LongTensor([len(indexed)]).to(device)
        tensor = torch.LongTensor(indexed).to(device)
        tensor = tensor.unsqueeze(1)
        prediction = F.sigmoid(model(tensor, length_tensor))
        return prediction.item()
