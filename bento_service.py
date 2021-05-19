from bentoml import BentoService, api, artifacts, env
from bentoml.adapters import JsonInput
from bentoml.frameworks.keras import KerasModelArtifact
from bentoml.frameworks.transformers import TransformersModelArtifact
from bentoml.frameworks.pytorch import PytorchModelArtifact
from bentoml.service.artifacts.common import PickleArtifact
from tensorflow.keras.preprocessing import sequence, text
import torch

from data import preprocess


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
        input_data = [raw[: n + 1] for n in range(len(raw))]
        input_data = sequence.pad_sequences(input_data, maxlen=100, padding="post")
        return self.artifacts.model.predict(input_data, verbose=1)


@env(infer_pip_packages=True)
@artifacts([TransformersModelArtifact("bert")])
class TransformersService(BentoService):
    @api(input=JsonInput(), batch=False)
    def predict(self, parsed_json):
        src_text = parsed_json.get("text")
        model = self.artifacts.bert.get("model")
        tokenizer = self.artifacts.bert.get("tokenizer")
        with torch.no_grad():
            tokens = tokenizer.tokenize(src_text)
            tokens = ['[CLS]'] + tokens + ['[SEP]']
            tokens_ids = tokenizer.convert_tokens_to_ids(tokens)
            seq = torch.tensor(tokens_ids)
            seq = seq.unsqueeze(0)
            attn_mask = (seq != 0).long()
            logit = model(seq, attn_mask)
            prob = torch.sigmoid(logit.unsqueeze(-1))
            prob = prob.item()
            soft_prob = prob > 0.5
            if soft_prob == 1:
                return prob
            else:
                return 1 - prob
