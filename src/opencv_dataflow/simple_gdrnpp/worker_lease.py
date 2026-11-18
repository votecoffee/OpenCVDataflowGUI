"""Explicit shared ownership across the host's execution and display graph copies."""

from threading import Lock


class WorkerLease:
    def __init__(self, client):
        self.client = client
        self._owners = 1
        self._lock = Lock()

    def acquire(self):
        with self._lock:
            if not self._owners:
                raise RuntimeError("The completed worker session has closed; run again")
            self._owners += 1
        return self

    def release(self):
        with self._lock:
            if not self._owners:
                raise RuntimeError("Worker lease released twice")
            self._owners -= 1
            close = not self._owners
        if close:
            self.client.close()
