import threading


# asyncio.Future implementation
class FutureImpl:
    def __init__(self, task_id, task_size, future_cache_ref):
        self._id = task_id
        self._size = task_size
        self._future_cache_ref = future_cache_ref
        self._outputs = []
        self._finish_event = threading.Event()

    def result(self, timeout=None):
        if self._size == 0:
            self._finish_event.set()
            return []
        finished = self._finish_event.wait(timeout)
        if not finished:
            raise TimeoutError(f"task: {self._id} timeout")

        # remove from future cache
        future_cache = self._future_cache_ref()
        if future_cache is not None:
            del future_cache[self._id]

        self._outputs.sort(
            key=lambda i: i[0]
        )  # we will sort by req_id [[req_id, output], ...]
        return [
            i[1] for i in self._outputs
        ]  # we will then return output from our batch result

    def done(self):
        if self._finish_event.is_set():
            return True

    def _append_result(self, req_id, output):
        self._outputs.append((req_id, output))
        if len(self._outputs) > self._size:
            self._finish_event.set()


class FutureCache(dict):
    """A dict for weakref only"""

    pass
