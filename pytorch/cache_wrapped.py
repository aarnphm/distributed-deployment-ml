import functools
import typing as t

import numpy as np
import pandas as pd
import tensorflow as tf
import torch
import transformers
import xgboost as xgb
from bentoml.pytorch import PyTorchModel
from bentoml.tensorflow import TensorflowModel
from bentoml.xgboost import XgBoostModel
from bentoml.transformers import TransformersModel, TransformersModelInput

TF2 = tf.__version__.startswith('2')


class Runnable:
    def __init__(self, batch=True, batch_dim=0, method_name=None, model_path=None):
        if not batch and hasattr(self, 'run_batch'):
            raise AttributeError(assert_msg.format(classname=self.__class__.__name__))
        self._batch = batch
        self._batch_dim = batch_dim

        self._method_name = method_name
        self._model_path = model_path
        self._method_call = None
        self._model = None

    def __getattribute__(self, item):
        _inherited_member = object.__getattribute__(self, item)
        if item.startswith('run'):

            def wrapped_run_method(*args, **kw):
                if self._model is None:
                    print(f'invoking setup while calling {item}')
                    self.setup()
                if self._method_call is None:
                    self._method_call = getattr(self._model, self._method_name)
                return _inherited_member(*args, **kw)

            return wrapped_run_method
        elif item == 'setup':

            @functools.lru_cache(maxsize=1)
            def wrapped_setup(*args, **kw):
                return _inherited_member(*args, **kw)

            return wrapped_setup
        else:
            return _inherited_member

    def setup(self):
        """Setup implementation"""

    def run(self, *args, **kwargs):
        ...

    def run_batch(self, *args, **kwargs):
        ...


class TensorflowRunnable(Runnable):
    def __init__(self, model_path, method_name="predict", device_id='CPU:0', batch_dim=0, batch=True):
        super(TensorflowRunnable, self).__init__(batch, batch_dim, method_name=method_name, model_path=model_path)
        from tensorflow.python.client import device_lib
        if debug:
            tf.debugging.set_log_device_placement(True)

        devs = device_lib.list_local_devices()
        assert any(device_id in d.name for d in devs)
        # self.devices is a TensorflowEagerContext
        self._device = tf.device(device_id)
        if TF2:
            if 'GPU' in device_id:
                # limit runner to specific gpu (GPU:0) (only available with 2.0+)
                tf.config.set_visible_devices(device_id, 'GPU')
        else:
            self._method_call = self.model.signatures['serving_default']

    def setup(self):
        self._model = TensorflowModel.load(self.model_path)

    def run(self, input_data: t.Union[np.ndarray, tf.Tensor]) -> t.Union[np.ndarray, tf.Tensor]:
        with tf.Session(
                config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)) as sess, self.device:
            sess.run(tf.global_variables_initializer())
            if not TF2:
                sess.run(self.model, {"input": input_tensor})
            res = self.method_call(input_tensor)
            return res if TF2 else res['prediction']

    def run_batch(self, input_data: t.Union[np.ndarray, tf.Tensor]) -> t.Union[np.ndarray, tf.Tensor]:
        with tf.Session(
                config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)) as sess, self.device:
            sess.run(tf.global_variables_initializer())
            if not TF2:
                sess.run(self.model, {"input": input_data})
            res = self.method_call(input_tensor)
            return res if TF2 else res['prediction']


class PyTorchRunnable(Runnable):
    def __init__(
            self,
            model_path,
            device_id="cuda:0",
            method_name="__call__",
            batch_dim=0,
            batch=True,
    ):
        super(PyTorchRunnable, self).__init__(
            batch=batch,
            batch_dim=batch_dim,
            method_name=method_name,
            model_path=model_path,
        )
        self._device = torch.device(device_id if torch.cuda.is_available() else 'cpu')

    def setup(self):
        self._model = PyTorchModel.load(self._model_path).to(self._device)

    def run(self, *inputs, **kw) -> torch.Tensor:
        with torch.no_grad():
            return self._method_call(*inputs, **kw)

    def run_batch(self, *inputs, **kw) -> torch.Tensor:
        with torch.no_grad():
            return self._method_call(*inputs, **kw)


class XgboostRunnable(Runnable):
    def __init__(
            self,
            model_path,
            method_name="predict",
            batch_dim=0,
            batch=True,
    ):
        super(XgboostRunnable, self).__init__(
            batch=batch,
            batch_dim=batch_dim,
            method_name=method_name,
            model_path=model_path,
        )

    def setup(self):
        self._model = XgboostModel.load(self._model_path)

    def run(self, input_data: t.Union[xgb.DMatrix, pd.DataFrame, np.array]) -> np.array:
        if not isinstance(input_data, xgb.DMatrix):
            dm = xgb.DMatrix(input_data)
        else:
            dm = input_data
        res = self._method_call(dm)
        return np.asarray(res)

    def run_batch(self, input_data: t.Union[xgb.DMatrix, pd.DataFrame, np.array]) -> np.array:
        if not isinstance(input_data, xgb.DMatrix):
            dm = xgb.DMatrix(input_data)
        else:
            dm = input_data
        res = self._method_call(dm)
        return np.asarray(res)


class TransformersRunnable(Runnable):
    def __init__(
            self,
            tasks: str,
            model: t.Union[str, t.Dict[str, t.Union[TransformersModelInput, transformers.AutoTokenizer]]],
            batch_dim=0,
            batch=True,
    ):
        super(TransformersRunnable, self).__init__(
            batch=batch,
            batch_dim=batch_dim,
        )
        self._tasks = tasks
        self._model_identifier = model
        self._pipeline: transformers.pipelines.Pipeline

    def load_pipeline(self, tasks):
        transformers_dict = TransformersModel.load(self._model_identifier)
        model, tokenizer = transformers_dict.values()
        transformers.pipelines.check_task(tasks)
        runnable_pipeline = transformers.pipeline(tasks, model=model, tokenizer=tokenizer)
        return runnable_pipeline

    def setup(self):
        self._pipeline = self.load_pipeline(self._tasks)

    def run(self, *inputs, **kw):
        return self._pipeline(*inputs, **kw)

    def run_batch(self, *inputs, **kw):
        return self._pipeline(*inputs, **kw)


class GPTTransformers(TransformersRunnable):
    @staticmethod
    def load_pipeline(tasks='sentiment-analysis', **kwargs):
        print('loaded')
        super().load_pipeline(tasks)


def create_transformers_runnable(runnable_cls: TransformersRunnable):
    if not hasattr(runnable_cls, 'load_pipeline'):
        raise AttributeError
    return runnable_cls