import sys
import time

# Callback function for when error is called
_error_callback = None
_warn_callback = None

class TimeThis:
    def __init__(self, taskname: str, logfn=None):
        self.taskname = taskname
        if not logfn:
            self.logfn = lambda x: print(x)
        else:
            self.logfn = logfn

    def __enter__(self):
        self.logfn(f"Starting: {self.taskname}")
        self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        end = time.perf_counter()
        elapsed_ms = (end - self.start) * 1000
        self.logfn(f"Elapsed time for {self.taskname}: {elapsed_ms:.2f} ms")
        
class LogLevel:
    NONE = 0
    ERROR = 1
    WARN = 2
    INFO = 3
    DEBUG = 4
    TRACE = 5
    
    _instance = None
    _level = NONE

    @classmethod
    def setLevel(cls, level):
        """Sets the global log level."""
        cls._level = level

    @classmethod
    def getLevel(cls):
        """Returns the current log level."""
        return cls._level
    
def log(*args):
    """Prints log messages based on the log level."""
    if args[0] <= LogLevel.getLevel():
        log_message = " ".join(map(str, args[1:]))
        print(log_message)
        if _error_callback and args[0] == LogLevel.ERROR:
            _error_callback(log_message)
        if _warn_callback and args[0] == LogLevel.WARN:
            _warn_callback(log_message)

def error(*args, exception=None):
    """Prints error messages."""
    log_message = "ERROR: " + " ".join(map(str, args))
    print("ERROR:", log_message, file=sys.stderr)
    if exception and LogLevel.getLevel() >= LogLevel.DEBUG:
        # Print exception
        import traceback
        traceback.print_exception(type(exception), exception, exception.__traceback__)
    if _error_callback:
        _error_callback(log_message)

def register_error_callback(callback):
    """Registers a callback function for when error is called."""
    global _error_callback
    _error_callback = callback

def register_warn_callback(callback):
    """Registers a callback function for when warn is called."""
    global _warn_callback
    _warn_callback = callback

def fmt_time(t):
    if t > 1:
        return f"{t:.2f} seconds"
    elif t > 0.001:
        return f"{t*1000:.2f} ms"
    elif t > 0.000001:
        return f"{t*1000000:.2f} us"
    else:
        return f"{t*1000000000:.2f} ns"
    
def log_performance(func):
    from functools import wraps
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        log(LogLevel.DEBUG, f"{func.__name__} took {fmt_time(end_time - start_time)}")
        return result
    return wrapper

_performance_logs = {}
def get_performance_logs():
    global _performance_logs
    return _performance_logs

def record_performance(func):
    from functools import wraps
    global _performance_logs
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        log = _performance_logs.setdefault(func.__name__, [])
        log.append(end_time - start_time)
        return result
    return wrapper

def clear_performance_logs():
    global _performance_logs
    _performance_logs.clear()

def dump_performance_logs():
    global _performance_logs
    log(LogLevel.DEBUG, "Performance logs:")
    for func, times in _performance_logs.items():
        count = len(times)
        sum_time = sum(times)
        avg_time = sum_time / count
        log(LogLevel.DEBUG, f"    {func:<40} | {count:<6} calls | total {fmt_time(sum_time)} | avg {fmt_time(avg_time)}")