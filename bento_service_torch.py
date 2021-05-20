from bentoml import BentoService, api, artifacts, env
from bentoml.adapters import JsonInput
from bentoml.frameworks.pytorch import PytorchModelArtifact
import torch
import torch.nn.functional as F
from train_torch import TEXT
from model_torch import device


@env(infer_pip_packages=True)
@artifacts([PytorchModelArtifact("torchmodel")])
class PytorchService(BentoService):
    @api(input=JsonInput())
    def predict(self, parsed_json, min_len=5):
        src_text = parsed_json.get("text")
        model = self.artifacts.torchmodel.get("model")
        nlp = self.artifacts.torchmodel.get("tokenizer")
        model.eval()
        tokenized = [tok.text for tok in nlp.tokenizer(src_text)]
        if len(tokenized) < min_len:
            tokenized += ['<pad>'] * (min_len - len(tokenized))
        indexed = [TEXT.vocab.stoi[t] for t in tokenized]
        tensor = torch.LongTensor(indexed).to(device)
        tensor = tensor.unsqueeze(0)
        prediction = F.sigmoid(model(tensor))
        return prediction.item()
