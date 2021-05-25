import torch
from bentoml import BentoService, api, env, artifacts
from bentoml.frameworks.onnx import OnnxModelArtifact
from bentoml.service.artifacts.pickle import PickleArtifact
from bentoml.adapters import JsonInput, JsonOutput
from helpers import get_pipeline

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


@env(infer_pip_packages=True, pip_packages=['onnxruntime-gpu'])
@artifacts(
    [OnnxModelArtifact('model', backend='onnxruntime-gpu'), PickleArtifact('tokenizer'), PickleArtifact('vocab')])
class OnnxService(BentoService):
    def __init__(self):
        super().__init__()
        self.news_label = {1: 'World',
                           2: 'Sports',
                           3: 'Business',
                           4: 'Sci/Tec'}

    @api(input=JsonInput(), output=JsonOutput())
    def predict(self, parsed_json):
        text_pipeline, _ = get_pipeline(self.artifacts.tokenizer, self.artifacts.vocab)
        sentence = parsed_json.get('text')
        text = torch.tensor(text_pipeline(sentence))
        # offset = torch.tensor([0])
        tensor_name = self.artifacts.model.get_inputs()[0].name
        # offset_name = self.artifacts.model.get_inputs()[1].name
        onnx_inputs = {tensor_name: to_numpy(text)}
        return self.artifacts.model.run(None, onnx_inputs)