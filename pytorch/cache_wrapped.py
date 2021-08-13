import pandas as pd
import numpy as np
import torch
import functools
import xgboost as xgb
from bentoml.pytorch import PyTorchModel


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
                # implement caching setup
                if self._model is None:
                    print(f'invoking setup while calling {item}')
                    self.setup()
                if self._method_call is None:
                    self._method_call = getattr(self.model, self._method_name)
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


class PyTorchModelRunnable(Runnable):
    def __init__(
        self,
        model_path,
        device_id="cuda:0",
        method_name="__call__",
        batch_dim=0,
        batch=True,
    ):
        super(PyTorchModelRunnable, self).__init__(
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


class XgboostModelRunnable(Runnable):
    def __init__(
        self,
        model_path,
        method_name="predict",
        batch_dim=0,
        batch=True,
    ):
        super(XgboostModelRunnable, self).__init__(
            batch=batch,
            batch_dim=batch_dim,
            method_name=method_name,
            model_path=model_path,
        )
    def run(self, input_data: pd.DataFrame) -> np.array:
        dm = xgb.DMatrix(input_data)
        res = self._method_call(dm)
        return np.asarray()



