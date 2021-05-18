import os
from typing import List, Union

import torch
import torch.nn as nn
from transformers import BertPreTrainedModel, BertModel, AutoTokenizer

from args import model_name_or_path


class Manager(object):
    """
    Manager is a way to lazy initialize our model and allocate set GPU to given model.

    Example:
        from transformers import BertPreTrainedModel, BertModel
        # create your model here

        class CustomBert(Manager):

            def setup_model(self):
                self.model = Model()

            def predict(self, batch):
                return self.model.predict(batch)
    """

    def __init__(self, model=None, gpu_id=None):
        self.model = model.from_pretrained(model_name_or_path)
        self.gpu_id = gpu_id
        self.set_gpu_id(self.gpu_id)

    @staticmethod
    def set_gpu_id(gpu_id=None):
        if gpu_id is None:
            raise ValueError("gpu_id shouldn't be None")
        # We can set the gpu_id for our model via CUDA_VISIBLE_DEVICES
        # https://stackoverflow.com/a/37901914/8643197
        # https://github.com/pytorch/pytorch/issues/20606
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    def predict(self, *args, **kwargs) -> Union[str, List, int]:
        raise NotImplementedError


class BertForSentimentClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # The classification layer that takes the [CLS] representation and outputs the logit
        self.cls_layer = nn.Linear(config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        """
        :param input_ids: Tensor of shape [B, T] containing token ids of sequences
        :param attention_mask: Tensor of shape [B, T] containing attention masks to be used to avoid contribution of
               PAD tokens.
        :return: logits
        """
        # Feed the input to Bert model to obtain outputs
        outs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # Obtain the representations of [CLS] heads
        cls_reps = outs.last_hidden_state[:, 0]
        cls_reps = self.dropout(cls_reps)
        logits = self.cls_layer(cls_reps)
        return logits


class ManagedBertModel(Manager):
    def __init__(self, model=BertForSentimentClassification):
        super(ManagedBertModel, self).__init__(model)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    def predict(self, batch) -> List:
        batch_outputs = []
        for text in batch:
            with torch.no_grad():
                tokens = self.tokenizer.tokenize(text)
                tokens = ['[CLS]'] + tokens + ['[SEP]']
                tokens_ids = self.tokenizer.convert_tokens_to_ids(tokens)
                seq = torch.tensor(tokens_ids)
                seq = seq.unsqueeze(0)
                attn_mask = (seq != 0).long()
                logit = self.model(seq, attn_mask)
                prob = torch.sigmoid(logit.unsqueeze(-1))
                prob = prob.item()
                batch_outputs.append(prob)
        return batch_outputs
