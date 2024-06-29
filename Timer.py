import time

class TimerError(Exception):
    """A custom exception used to report errors in use of Timer class"""

class Timer:
    def __init__(self):
        self._start_time = None # Initialize start time to None
        self._elapsed_time = 0 # Initialize elapsed time to 0

    def start(self):
        if self._start_time is None:
            self._start_time = time.perf_counter() # Start the timer using perf_counter() from time module
        return self._start_time

    def stop(self):
        if self._start_time is None:
            raise TimerError("Timer is not running. Use .start() to start it")  # Raise an error if timer is not running
        self._elapsed_time = time.perf_counter() - self._start_time # Calculate elapsed time since start
        self._start_time = None # Reset start time to None
        return self._elapsed_time # Return elapsed time

    def reset(self):
        self._start_time = None # Reset start time to None
        self._elapsed_time = 0 # Reset elapsed time to 0

    def elapsed_time(self):
        if self._start_time is not None:
            return time.perf_counter() - self._start_time # Return current elapsed time if timer is runnin
        return self._elapsed_time # Otherwise, return stored elapsed time
 
    @staticmethod    
    def current_timer():
        return time.perf_counter() # Static method to return the current value of the timer using perf_counter()