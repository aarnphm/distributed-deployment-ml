import torch
from bentoml import BentoService, api, env, artifacts
from bentoml.frameworks.onnx import OnnxModelArtifact
from bentoml.service.artifacts.pickle import PickleArtifact
from bentoml.adapters import JsonInput, JsonOutput
from torch.train import vocab
from torch.model import device


@env(infer_pip_packages=True)
@artifacts([OnnxModelArtifact('model', backend='onnxruntime-gpu'), PickleArtifact('tokenizer')])
class OnnxService(BentoService):
    @api(input=JsonInput(), batch=False, output=JsonOutput())
    def predict(self, parsed_json):
        sentence = parsed_json.get("text")
        tokenized = [tok.text for tok in self.artifacts.tokenizer.tokenizer(sentence)]
        indexed = [vocab.stoi[t] for t in tokenized]
        length = torch.LongTensor([len(indexed)])
        tensor = torch.LongTensor(indexed).to(device).unsqueeze(1)
        tensor_name = self.artifacts.model.get_inputs()[0].name
        length_name = self.artifacts.model.get_inputs()[1].name
        onnx_inputs = {
            tensor_name: tensor,
            length_name: length,
        }
        return self.artifacts.model.run(None, onnx_inputs)