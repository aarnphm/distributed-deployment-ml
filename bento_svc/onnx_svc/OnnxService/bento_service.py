import torch
from bentoml import BentoService, api, env, artifacts
from bentoml.adapters import JsonInput, JsonOutput
from bentoml.frameworks.onnx import OnnxModelArtifact
from bentoml.service.artifacts.pickle import PickleArtifact
from onnxruntime.capi.onnxruntime_pybind11_state import InvalidArgument

from helpers import get_pipeline

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def to_numpy(tensor):
    return tensor.detach().cpu().clone().numpy() if tensor.requires_grad else tensor.cpu().clone().numpy()


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

    def classify_categories(self, sentence):
        text_pipeline, _ = get_pipeline(self.artifacts.tokenizer, self.artifacts.vocab)
        text = to_numpy(torch.tensor(text_pipeline(sentence)).to(device))
        tensor_name = self.artifacts.model.get_inputs()[0].name
        output_name = self.artifacts.model.get_outputs()[0].name
        onnx_inputs = {tensor_name: text}
        print(f'providers: {self.artifacts.model.get_providers()}')

        try:
            r = self.artifacts.model.run([output_name], onnx_inputs)[0]
            return r.argmax(1).item() + 1
        except (RuntimeError, InvalidArgument) as e:
            print(f"ERROR with shape: {onnx_inputs[tensor_name].shape} - {e}")

    @api(input=JsonInput(), output=JsonOutput())
    def predict(self, parsed_json):
        sentence = parsed_json.get('text')
        return {'categories': self.news_label[self.classify_categories(sentence)]}