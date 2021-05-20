import os
from typing import List, Union

from workers.args import model_name_or_path


class Manager(object):
    """
    Manager is a way to lazy initialize our tfmodel and allocate set GPU to given tfmodel.

    Example:
        from transformers import BertPreTrainedModel, BertModel
        # create your tfmodel here

        class CustomBert(Manager):

            def setup_model(self):
                self.tfmodel = TorchModel()

            def predict(self, batch):
                return self.tfmodel.predict(batch)
    """

    def __init__(self, model=None, gpu_id=None):
        self.model = model.from_pretrained(model_name_or_path)
        self.gpu_id = gpu_id
        self.set_gpu_id(self.gpu_id)

    @staticmethod
    def set_gpu_id(gpu_id=None):
        if gpu_id is None:
            raise ValueError("gpu_id shouldn't be None")
        # We can set the gpu_id for our tfmodel via CUDA_VISIBLE_DEVICES
        # https://stackoverflow.com/a/37901914/8643197
        # https://github.com/pytorch/pytorch/issues/20606
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    def predict(self, *args, **kwargs) -> Union[str, List, int]:
        raise NotImplementedError
