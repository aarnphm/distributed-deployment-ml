import logging
import multiprocessing
import os
import threading
import time
import uuid
import weakref
from queue import Empty
from typing import List, Callable, Union, Tuple, Optional

import torch.nn as nn

from future_impl import FutureCache, FutureImpl
from manager import Manager

TIMEOUT = 1
WORKER_TIMEOUT = 20
SLEEP = 1e-2
logger = logging.getLogger(__name__)
logger.setLevel("DEBUG")


class Base:
    # Base shouldn't get exposed to others to call
    def __init__(self, *args, **kwargs):
        super().__init__()
        self._task_id = 0
        self._client_id = str(uuid.uuid4())
        self._future_cache = FutureCache()  # {task_id: future}
        self._worker_timeout = kwargs.get("worker_time_out", WORKER_TIMEOUT)
        self.back_thread = threading.Thread(target=self._loop_get_result, name="thread_get_result")
        self.back_thread.daemon = True
        self.lock = threading.Lock()

    def _delay_setup(self):
        self.back_thread.start()

    def _send_req(self, task_id, request_id, model_input):
        raise NotImplementedError

    def _recv_resp(self, timeout=TIMEOUT):
        raise NotImplementedError

    def destroy_worker(self):
        raise NotImplementedError

    def _loop_get_result(self):
        logger.info("start _loop_get_result")
        while True:
            message = self._recv_resp(timeout=TIMEOUT)
            if message:
                task_id, req_id, item = message
                future = self._future_cache[task_id]
                future._append_result(req_id, item)
            else:
                time.sleep(SLEEP)

    def _input(self, batch: List) -> int:
        # input a batch -> distribute to each message queue -> return given task_id
        self.lock.acquire()
        task_id = self._task_id
        self._task_id += 1
        self.lock.release()
        req_id = 0

        future = FutureImpl(task_id, len(batch), weakref.ref(self._future_cache))
        self._future_cache[task_id] = future

        for model_input in batch:
            self._send_req(task_id, req_id, model_input)
            req_id += 1
        return task_id

    def _output(self, task_id: int) -> List:
        return self._future_cache[task_id].result(self._worker_timeout)

    def submit(self, batch):
        task_id = self._input(batch)
        return self._future_cache[task_id]

    def predict(self, batch):
        task_id = self._input(batch)
        res = self._output(task_id)
        assert len(batch) == len(res), f"input batch size {len(batch)} should equal output batch size {len(res)}"
        return res


class Worker:
    def __init__(self, predict_fn, batch_size, max_latency, *args, **kwargs):
        super().__init__()
        if not callable(predict_fn):
            raise ValueError("predict function is not callable")
        self._pid = os.getpid()
        self._predict = predict_fn
        self._batch_size = batch_size
        self._max_latency = max_latency
        self._destroy_event = kwargs.get('destroy_event', None)

    def _recv_req(self, timeout=TIMEOUT):
        raise NotImplementedError

    def _send_resp(self, client_id, task_id, req_id, model_input):
        raise NotImplementedError

    def run(self, *args, **kwargs):
        self._pid = os.getpid()
        logger.info(f"[gpu worker {self._pid}] {self} running")

        while True:
            handler = self._start_once()
            if self._destroy_event and self._destroy_event.is_set():
                break
            if not handler:
                # sleep if not data is handled
                time.sleep(SLEEP)
        logger.info(f"[gpu worker {self._pid}] {self} shutdown")

    def model_predict(self, batch):
        res = self._predict(batch)
        assert len(batch) == len(res), f"input batch size {len(batch)} should equal output batch size {len(res)}"
        return res

    def _start_once(self) -> int:
        batch = []
        start_time = time.time()
        for i in range(self._batch_size):
            try:
                item = self._recv_req(timeout=self._max_latency)
            except TimeoutError:
                break
            else:
                batch.append(item)
            if (time.time() - start_time) > self._max_latency:
                break  # break when total batch time exceeds our max latency
        if not batch:
            return 0

        model_inputs = [i[3] for i in batch]
        model_outputs = self.model_predict(model_inputs)

        for i, item in enumerate(batch):
            client_id, task_id, request_id, _ = item
            self._send_resp(client_id, task_id, request_id, model_outputs[i])

        batch_size = len(batch)
        logger.info(f"[gpu worker {self._pid}] _start_once batch_size: {batch_size} start_at: {start_time} spend: {time.time()-start_time:.4f}"
        return batch_size


class DispatchWorker(Worker):
    def __init__(self, predict_fn_or_model: Union[nn.Module, Callable],
                 batch_size, max_latency, req_queue, resp_queue, model_args,
                 model_kwargs, *args, **kwargs):
        super(DispatchWorker, self).__init__(predict_fn_or_model, batch_size, max_latency, *args, **kwargs)
        self._req_queue = req_queue
        self._resp_queue = resp_queue
        self._model_args = model_args or []
        self._model_kwargs = model_kwargs or {}

    def run(self, gpu_id=None, ready_event=None, destroy_event=None):
        if isinstance(self._predict, type) and issubclass(self._predict, Manager):
            model_class = self._predict
            logger.info(f"[gpu worker {os.getpid()}] init model on cuda:{gpu_id}")
            self._model = model_class(gpu_id)
            self._model.setup_model(*self._model_args, **self._model_kwargs)
            logger.info(f"[gpu worker {os.getpid()}] init model on cuda:{gpu_id}")
            self._predict = self._model.predict
        if ready_event:
            ready_event.set()
        if destroy_event:
            self._destroy_event = destroy_event
        super().run()

    def _recv_req(self, timeout=TIMEOUT):
        try:
            item = self._req_queue.get(timeout)
        except Empty:
            raise TimeoutError
        else:
            return item

    def _send_resp(self, client_id, task_id, req_id, model_input):
        self._resp_queue.put((task_id, req_id, model_input))


class Dispatcher(Base):
    def __init__(self, predict_fn_or_model, batch_size, max_latency=0.1,
                 worker_num: Optional[int]=1, cuda_devices: Optional[Tuple[int]]=None,
                 model_args=None, model_kwargs=None, wait_for_worker_ready=False,
                 mp_start_method='spawn', worker_timeout=WORKER_TIMEOUT):
        super(Dispatcher, self).__init__(worker_timeout)
        self.worker_num = worker_num
        self.cuda_devices = cuda_devices
        self.mp = multiprocessing.get_context(mp_start_method)
        self._input_queue = self.mp.Queue()
        self._output_queue = self.mp.Queue()
        self._worker = DispatchWorker(predict_fn_or_model, batch_size, max_latency,
                                      self._input_queue, self._output_queue, model_args, model_kwargs)
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
                gpu_id = self.cuda_devices[i%len(self.cuda_devices)]
                args = (gpu_id, ready_event, destroy_event)
            else:
                args = (None, ready_event, destroy_event)
            p = self.mp.Process(target=self._worker.run, args=args, name="dispatcher_worker", daemon=True)
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




