import time

class TimerError(Exception):
    """A custom exception used to report errors in use of Timer class"""

class Timer:
    def __init__(self):
        self._start_time = None
        self._elapsed_time = 0

    def start(self):
        if self._start_time is None:
            self._start_time = time.perf_counter()
        return self._start_time

    def stop(self):
        if self._start_time is None:
            raise TimerError("Timer is not running. Use .start() to start it")
        self._elapsed_time = time.perf_counter() - self._start_time
        self._start_time = None
        return self._elapsed_time

    def reset(self):
        self._start_time = None
        self._elapsed_time = 0

    def elapsed_time(self):
        if self._start_time is not None:
            return time.perf_counter() - self._start_time
        return self._elapsed_time

    @staticmethod    
    def current_timer():
        return time.perf_counter()