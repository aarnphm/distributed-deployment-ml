from typing import List
import os


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

    def __init__(self, gpu_id=None):
        self.model = None
        self.gpu_id = gpu_id
        self.set_gpu_id(self.gpu_id)

    @staticmethod
    def set_gpu_id(gpu_id=None):
        if gpu_id is None:
            raise ValueError("gpu_id shouldn't be None")
        # We can set the gpu_id for our model via CUDA_VISIBLE_DEVICES
        # https://stackoverflow.com/a/37901914/8643197
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    def setup_model(self, *args, **kwargs):
        raise NotImplementedError

    def predict(self, batch: List) -> List:
        raise NotImplementedError
