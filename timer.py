from datetime import datetime, timedelta
from functools import wraps


class IntervalTimer:
    """Timer that can be used to run code at a specified interval.

    Args:
        **kwargs: Any argument that can be passed into datetime.timedelta


    Examples:
        Prints the current time every 10 seconds

        >>> timer = IntervalTimer(seconds=10)
        >>> while True:
        >>>      if timer.check_and_reset_if_ready():
        >>>            print(datetime.datetime.now())
        2023-03-27 15:40:34.768318
        2023-03-27 15:40:44.768328
        2023-03-27 15:40:54.768386
        2023-03-27 15:41:04.768367


        >>> timer=IntervalTimer(seconds=20)
        >>> @timer.decorate_with_timer
        >>> def myfn():
        >>>    print(datetime.datetime.now())
        >>> myfn()
        2023-03-27 15:40:34.768318
        2023-03-27 15:40:54.768328
        2023-03-27 15:40:14.768386
        2023-03-27 15:41:34.768367


    """

    def __init__(self, **kwargs):
        self.next_call = datetime.now()
        self._interval = timedelta()
        self.interval = timedelta(**kwargs)

    @property
    def ready(self):
        return self.next_call < datetime.now()

    def reset(self):
        """Reset the timer."""
        self.next_call = datetime.now() + self.interval

    def check_and_reset_if_ready(self):
        """Returns and resets the timer if the timer is ready. Otherwise returns false."""
        ready = self.ready
        if ready:
            self.reset()
        return ready

    @property
    def interval(self):
        return self._interval

    @interval.setter
    def interval(self, value):
        if not self.ready:
            self.next_call += value - self._interval
        self._interval = value

    def decorate_with_timer(self, fn):
        @wraps(fn)
        def decorated_fun(*args, **kwargs):
            while True:
                if self.check_and_reset_if_ready():
                    fn(*args, **kwargs)

        return decorated_fun
