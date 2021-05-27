from bentoml import BentoService, api, artifacts, env
from bentoml.adapters import JsonInput, JsonOutput
from bentoml.frameworks.pytorch import PytorchModelArtifact
from bentoml.service.artifacts.pickle import PickleArtifact
from train import get_pipeline
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  ## cuda


@env(infer_pip_packages=True, pip_packages=['torchtext'])
@artifacts([PytorchModelArtifact("model"), PickleArtifact("tokenizer"), PickleArtifact("vocab")])
class PytorchService(BentoService):
    def __init__(self):
        super().__init__()
        self.news_label = {1: 'World',
                           2: 'Sports',
                           3: 'Business',
                           4: 'Sci/Tec'}

    def classify_categories(self, sentence):
        text_pipeline, _ = get_pipeline(self.artifacts.tokenizer, self.artifacts.vocab)
        with torch.no_grad():
            text = torch.tensor(text_pipeline(sentence)).to(device)
            offsets = torch.tensor([0]).to(device)
            # print(f"{text}\nsize:{text.size()}")
            output = self.artifacts.model(text, offsets=offsets)
            return output.argmax(1).item() + 1

    @api(input=JsonInput(), output=JsonOutput())
    def predict(self, parsed_json):
        label = self.classify_categories(parsed_json.get("text"))
        return {'categories': self.news_label[label]}