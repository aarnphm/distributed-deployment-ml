import multiprocessing
import os
from functools import wraps
from queue import Empty
from typing import List, Callable, Union, Tuple, Optional

import torch
import torch.nn as nn

from manager import Manager
from _worker import DispatcherBase, Worker

from args import logger, TIMEOUT, WORKER_TIMEOUT


class DispatchWorker(Worker):
    def __init__(
        self,
        predict_fn_or_model: Union[nn.Module, Callable],
        model_torch_or_tf: Union[nn.Module],
        batch_size,
        max_latency,
        req_queue,
        resp_queue,
        model_args,
        model_kwargs,
        *args,
        **kwargs,
    ):
        super(DispatchWorker, self).__init__(
            predict_fn_or_model, batch_size, max_latency, *args, **kwargs
        )
        if not isinstance(predict_fn_or_model, Callable) or isinstance(
            predict_fn_or_model, Manager
        ):
            raise RuntimeError(
                "cannot support current tfmodel. Remember to wraps Manager."
            )
        self._model = None
        self._model_torch = model_torch_or_tf
        self._req_queue = req_queue
        self._resp_queue = resp_queue
        self._model_args = model_args or []
        self._model_kwargs = model_kwargs or {}

    def run_forever(self, gpu_id=None, ready_event=None, destroy_event=None):
        if isinstance(self._predict, type) and issubclass(self._predict, Manager):
            model_class = self._predict
            if not torch.cuda.is_available():
                raise ValueError("cannot run dispatch worker without a nvidia gpus.")
            logger.info(f"[gpu worker {os.getpid()}] init tfmodel on cuda:{gpu_id}")
            self._model = model_class(model=self._model_torch, gpu_id=gpu_id)
            self._predict = self._model.predict
        if ready_event:
            ready_event.set()
        if destroy_event:
            self._destroy_event = destroy_event
        super().run_forever()

    def _recv_req(self, timeout=TIMEOUT):
        try:
            item = self._req_queue.get(timeout)
        except Empty:
            raise TimeoutError
        else:
            return item

    def _send_resp(self, client_id, task_id, req_id, model_input):
        self._resp_queue.put((task_id, req_id, model_input))


class Dispatcher(DispatcherBase):
    def __init__(
        self,
        predict_fn_or_model,
        model_torch_or_tf,
        batch_size,
        max_latency=0.1,
        worker_num: Optional[int] = 1,
        cuda_devices: Optional[Tuple[int]] = None,
        model_args=None,
        model_kwargs=None,
        wait_for_worker_ready=False,
        mp_start_method='spawn',
        worker_timeout=WORKER_TIMEOUT,
        *args,
        **kwargs,
    ):
        super(Dispatcher, self).__init__(worker_timeout, *args, **kwargs)
        self.worker_num = worker_num
        self.cuda_devices = cuda_devices
        self.mp = multiprocessing.get_context(mp_start_method)
        self._input_queue = self.mp.Queue()
        self._output_queue = self.mp.Queue()
        self._worker = DispatchWorker(
            predict_fn_or_model,
            model_torch_or_tf,
            batch_size,
            max_latency,
            self._input_queue,
            self._output_queue,
            model_args,
            model_kwargs,
        )
        self._worker_ps = []
        self._worker_ready_events = []
        self._worker_destroy_events = []
        self._setup_gpu_worker()
        if wait_for_worker_ready:
            self._wait_for_worker_ready()
        self._delay_setup()

    def _setup_gpu_worker(self):
        for i in range(self.worker_num):
            ready_event = self.mp.Event()
            destroy_event = self.mp.Event()
            if self.cuda_devices is not None:
                gpu_id = self.cuda_devices[i % len(self.cuda_devices)]
                args = (gpu_id, ready_event, destroy_event)
            else:
                args = (None, ready_event, destroy_event)
            p = self.mp.Process(
                target=self._worker.run_forever,
                args=args,
                name="dispatcher_worker",
                daemon=True,
            )
            p.start()
            self._worker_ps.append(p)
            self._worker_ready_events.append(ready_event)
            self._worker_destroy_events.append(destroy_event)

    def _wait_for_worker_ready(self, timeout=None):
        if timeout is None:
            timeout = self._worker_timeout
        # wait for all worker init
        for i, e in enumerate(self._worker_ready_events):
            ready = e.wait(timeout)
            logger.info(f'gpu worker: {i} ready state: {ready}')

    def _send_req(self, task_id, request_id, model_input):
        self._input_queue.put((0, task_id, request_id, model_input))

    def _recv_resp(self, timeout=TIMEOUT):
        try:
            message = self._output_queue.get(timeout=timeout)
        except Empty:
            message = None
        return message

    def destroy_worker(self):
        for e in self._worker_destroy_events:
            e.set()
        for p in self._worker_ps:
            p.join(timeout=self._worker_timeout)
            if p.is_alive():
                raise TimeoutError("worker_process destroy timeout")
        logger.info("workers destroyed")


def dispatcher(
    predict_fn_or_model: Optional[Callable] = None,
    model_torch_or_tf: Optional[Callable] = None,
    batch_size: Optional[int] = None,
    max_latency: Optional[Union[int, float]] = 0.1,
    worker_num: Optional[int] = 1,
    cuda_devices: Optional[Tuple[int]] = None,
    model_args: Optional[List[str]] = None,
    model_kwargs: Optional[List[str]] = None,
    wait_for_worker_ready: Optional[bool] = False,
    mp_start_method: Optional[str] = 'spawn',
    worker_timeout: Optional[int] = WORKER_TIMEOUT,
):
    def decorator(model_fn: Callable) -> Callable:
        @wraps(predict_fn_or_model)
        def wrapper(*args, **kwargs):
            return Dispatcher(
                model_fn,
                model_torch_or_tf,
                batch_size,
                max_latency,
                worker_num=worker_num,
                cuda_devices=cuda_devices,
                model_args=model_args,
                model_kwargs=model_kwargs,
                wait_for_worker_ready=wait_for_worker_ready,
                mp_start_method=mp_start_method,
                worker_timeout=worker_timeout,
                *args,
                **kwargs,
            )

        return wrapper

    return (
        decorator(predict_fn_or_model) if callable(predict_fn_or_model) else decorator
    )


def serve_gpu(model: Optional[nn.Module] = None, gpu_id: Optional[int] = 0):
    """
    serve_gpu is a decorator that enables users to enable users to run_forever inference on GPU  with given framework

    :param model: our tfmodel instance, using Pytorch as default, (will add tf support beyond)
    :param gpu_id: # of gpu instance, default to 1
    :return:
    """

    def decorator(fn: Callable) -> Callable:
        @wraps(fn)
        def wrapper(model_instance, *args, **kwargs):
            device = torch.device(
                f'cuda:{gpu_id}' if torch.cuda.is_available() else "cpu"
            )
            model_instance = model_instance.to(device)
            print(device)
            model_instance.eval()
            return model_instance

        # [i for i in tfmodel.state_dict()]
        print(f'decorating {fn} with tfmodel {model.modules()} and GPU: {gpu_id}')
        if not isinstance(model, nn.Module):
            raise RuntimeError("tfmodel is not of type nn.Module")
        return wrapper(model)

    return decorator
