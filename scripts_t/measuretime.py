from functools import wraps
from time import time


def measure(method):
    @wraps(method)
    def wrapper(*args, **kwargs):
        s = time()
        method(*args, **kwargs)
        e = time()
        mins, secs = divmod(e - s, 60)
        print(f"Total time of {method.__name__}: {mins:02.0f}:{secs:05.2f}s")

    return wrapper

