from bentoml import BentoService, api, artifacts, env
from bentoml.adapters import JsonInput
from bentoml.frameworks.keras import KerasModelArtifact
from bentoml.service.artifacts.common import PickleArtifact
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import text_to_word_sequence

from data import preprocess


@env(pip_packages=['scikit-learn', 'pandas', 'tensorflow'], docker_base_image="aarnphm/model-server:0.13.0-python3.7-slim-cudnn")
@artifacts([KerasModelArtifact('model'), PickleArtifact('tokenizer')])
class TensorflowService(BentoService):
    def word_to_index(self, word):
        if word in self.artifacts.tokenizer and self.artifacts.tokenizer[word] <= 5000:
            return self.artifacts.tokenizer[word]
        else:
            return self.artifacts.tokenizer["<OOV>"]

    def preprocessing(self, text_str):
        proc = text_to_word_sequence(preprocess(text_str))
        tokens = list(map(self.word_to_index, proc))
        return tokens

    @api(input=JsonInput())
    def predict(self, parsed_json):
        raw = self.preprocessing(parsed_json['text'])
        input_data = [raw[: n + 1] for n in range(len(raw))]
        input_data = pad_sequences(input_data, maxlen=100, padding="post")
        return self.artifacts.model.predict(input_data)
