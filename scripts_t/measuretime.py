from functools import wraps
from time import time
from datetime import datetime


def measure(method):
    @wraps(method)
    def wrapper(*args, **kwargs):
        mname = method.__name__
        s = time()
        print(f"Start  {mname} at: {datetime.now()}")
        method(*args, **kwargs)
        e = time()
        mins, secs = divmod(e - s, 60)
        print(f"Finish {mname} at: {datetime.now()}")
        print(f"Total time of {mname}: {mins:02.0f}:{secs:05.2f}s")

    return wrapper

