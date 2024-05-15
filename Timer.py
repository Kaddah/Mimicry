import time

class TimerError(Exception):
    """A custom exception used to report errors in use of Timer class"""

class Timer:
    def __init__(self):
        self._start_time = None
        self._elapsed_time = 0

    def start(self):
        #Start a new timer
        #if self._start_time is not None:
            #raise TimerError(f"Timer is running. Use .stop() to stop it")

        self._start_time = time.perf_counter()
        return self._start_time

    def stop(self):
        #Stop the timer, and report the elapsed time
        if self._start_time is None:
            raise TimerError(f"Timer is not running. Use .start() to start it")

        self._elapsed_time = time.perf_counter() - self._start_time
        self._start_time = None
        #print(f"elapsed time: {elapsed_time:0.4f} seconds")
        #return elapsed_time
    
    def reset(self):
        # Reset the timer
        self._start_time = 0
        self._elapsed_time = 0

    def elapsed_time(self):
        # Get the elapsed time since start or since last reset
        if self._start_time is not None:
            self.elapsed = time.perf_counter() - self._start_time + self._elapsed_time
            print(f"Elapsed time: {self.elapsed}")
            return self.elapsed
        else:
            print("Elapsed time: 0 (Timer not started)")
            return self._elapsed_time if self._elapsed_time is not None else 0

    @staticmethod    
    def current_timer():
        print(f"Current Timer : {time.perf_counter()}")
        Timer._current_time = time.perf_counter()
        return Timer._current_time