import os
import threading
import time
import uuid
import weakref
from typing import List

from args import logger, TIMEOUT, WORKER_TIMEOUT, SLEEP
from _future_impl import FutureCache, FutureImpl


class DispatcherBase:
    # DispatcherBase is our base representation for all dispatcher class.
    def __init__(self, *args, **kwargs):
        super().__init__()
        self._task_id = 0
        self._client_id = str(uuid.uuid4())
        self._future_cache = FutureCache()  # {task_id: future}
        self._worker_timeout = kwargs.get("worker_time_out", WORKER_TIMEOUT)
        self.back_thread = threading.Thread(
            target=self._loop_get_result, name="thread_get_result"
        )
        self.back_thread.daemon = True
        self.lock = threading.Lock()

    def _delay_setup(self):
        self.back_thread.start()

    def _send_req(self, task_id, request_id, model_input):
        # we will then send given task_id and request_id from flask with given model_input
        # downstream to our Manger to handle inference
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
        assert len(batch) == len(
            res
        ), f"input batch size {len(batch)} should equal output batch size {len(res)}"
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

    def run_forever(self, *args, **kwargs):
        self._pid = os.getpid()
        logger.info(f"[gpu worker {self._pid}] {self} running")

        while True:
            handler = self._start_once()
            if self._destroy_event and self._destroy_event.is_set():
                break
            if not handler:
                # sleep if not data.py is handled
                time.sleep(SLEEP)
        logger.info(f"[gpu worker {self._pid}] {self} shutdown")

    def model_predict(self, batch):
        res = self._predict(batch)
        assert len(batch) == len(
            res
        ), f"input batch size {len(batch)} should equal output batch size {len(res)}"
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
        logger.info(
            f"[gpu worker {self._pid}] _start_once batch_size: {batch_size} start_at: {start_time} \
            spend: {time.time() - start_time:.4f}"
        )
        return batch_size
