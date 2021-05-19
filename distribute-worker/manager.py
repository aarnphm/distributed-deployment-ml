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
