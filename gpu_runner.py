from functools import wraps
from typing import Callable, Optional

import torch
import torch.nn as nn


def serve_gpu(model: Optional[nn.Module] = None,
              gpu_id: Optional[int] = 0):
    """
    serve_gpu is a decorator that enables users to enable users to run inference on GPU  with given framework

    :param model: our model instance, using Pytorch as default, (will add tf support beyond)
    :param gpu_id: # of gpu instance, default to 1
    :return:
    """

    def decorator(fn: Callable) -> Callable:
        @wraps(fn)
        def wrapper(model_instance, *args, **kwargs):
            print('before\n')
            device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else "cpu")
            model_instance = model_instance.to(device)
            model_instance.eval()
            print('after\n')
            return model_instance

        print(f'decorating {fn} with model {[i for i in model.state_dict()]} and GPU: {gpu_id}')
        if not isinstance(model, nn.Module) or model is None:
            raise EnvironmentError("model is not of type torch.nn.Module or given model is None.")
        return wrapper

    return decorator
