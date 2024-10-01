import atexit
import functools
import itertools
import multiprocessing
import random
import sys
import time
from typing import Any, Callable

import joblib
import sympy

from colors import colors

TEST_COUNT = 0


def test_decorator(func: Callable) -> Callable:
    functools.wraps(func)
    global TEST_COUNT
    TEST_COUNT += 1
    _test_count = TEST_COUNT

    def wrapper(*args, **kwargs) -> Any:
        print(
            f"{colors.fg.red}Running{colors.fg.cyan} {func.__name__} {colors.fg.red}[{colors.reset}{_test_count}/{TEST_COUNT}{colors.fg.red}]{colors.reset}"
        )
        func(*args, **kwargs)

    return wrapper


def timer(func: Callable) -> Callable:
    functools.wraps(func)

    def wrapper(*args, **kwargs) -> Any:
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"Execution time of {func.__name__}: {end - start:.4f} seconds")
        return result

    return wrapper


# Repeat
def repeat(_func=None, *, num_times=2):
    def decorator_repeat(func):
        @functools.wraps(func)
        def wrapper_repeat(*args, **kwargs):
            for _ in range(num_times):
                value = func(*args, **kwargs)
            return value

        return wrapper_repeat

    if _func is None:
        return decorator_repeat
    else:
        return decorator_repeat(_func)


@test_decorator
def test_repeat():
    @repeat
    def say_whee():
        print("Whee!")

    @repeat(num_times=3)
    def greet(name: str):
        print(f"Hello {name}")

    say_whee()
    greet("Laurent")


# The @atexit.register decorator is used to register a function to be executed at program termination.
# This function can be used to perform any task when the program is about to exit,
# whether it’s due to normal execution or an unexpected error.


@atexit.register
def exit_handler():
    print("\n Exiting the Decorator Program. Cleanup tasks can be performed here.")


# Class Decorator
class printargskwargs:
    def __init__(self, debug: bool = False):
        self._debug = debug

    def __call__(self, func: Callable):
        functools.wraps(func)

        def wrapper(*args, **kwargs):
            if self._debug:
                args_repr = [repr(a) for a in args]
                kwargs_repr = [f"{k}={repr(v)}" for k, v in kwargs.items()]
                signature = ", ".join(args_repr + kwargs_repr)
                print(f"{func.__name__}({signature})")
            return func(*args, **kwargs)

        return wrapper


class interval_timer:
    def __init__(self, seconds: int):
        self._seconds = seconds

    def __call__(self, func: Callable):
        functools.wraps(func)

        def wrapper(*args, **kwargs):
            while True:
                start = time.time()
                func(*args, **kwargs)
                end = time.time()
                time.sleep(self._seconds - (end - start))

        return wrapper


@test_decorator
def test_interval_timer():
    @interval_timer(seconds=5)
    def say_laurent():
        print("Laurent")

    say_laurent()


@test_decorator
def test_class_decorator():
    @printargskwargs(debug=True)
    def say_word(word: str, n_times: int) -> str:
        return word * n_times

    @printargskwargs()
    def _say_word(word: str, n_times: int) -> str:
        return word * n_times

    print(say_word("Laurent", n_times=4))
    print(_say_word("Laurent", 4))


# Caching
@functools.lru_cache(maxsize=500)
def fibonacci(n):
    return n if n < 2 else fibonacci(n - 1) + fibonacci(n - 2)


# Input Validation
def validate_inputs(*args_types):
    def decorator(func):
        def wrapper(*args, **kwargs):
            for arg, arg_type in zip(args, args_types):
                if not isinstance(arg, arg_type):
                    raise ValueError(
                        f"Invalid argument type for {arg}. Expected {arg_type}"
                    )
            return func(*args, **kwargs)

        return wrapper

    return decorator


@test_decorator
def test_validate_inputs():
    @validate_inputs(int, str)
    def example_function(age, name):
        print(f"Age: {age}, Name: {name}")

    # Usage
    example_function(25, "John")


# Retry
def retry(max_attempts):
    def decorator(func):
        functools.wraps(func)

        def wrapper(*args, **kwargs):
            attempts = 0
            while attempts < max_attempts:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    print(f"Attempt {attempts+1} failed with error: {e}")
                    attempts += 1
                    delay = 2**attempts + random.uniform(0, 1)
                    time.sleep(delay)
            raise Exception("Max retries reached")

        return wrapper

    return decorator


@test_decorator
def test_retry():
    @retry(max_attempts=3)
    def unstable_function():
        if random.random() < 0.7:
            raise ValueError("Something went wrong")

    # Usage
    unstable_function()


# Deprecated
import warnings


def deprecated(func):
    functools.wraps(func)

    def wrapper(*args, **kwargs):
        warnings.warn(
            f"Function {func.__name__} is deprecated.", category=DeprecationWarning
        )
        return func(*args, **kwargs)

    return wrapper


@test_decorator
def test_deprecated():
    @deprecated
    def old_function():
        return "This is an old function"

    # Usage
    result = old_function()


# Parallel Computing


# MultiProcessing Version
def parallel_mp(
    func=None, args=(), merge_func=lambda x: x, parallelism=multiprocessing.cpu_count()
):
    def decorator(func: Callable):
        functools.wraps(func)

        def inner(*args, **kwargs):
            results = joblib.Parallel(n_jobs=parallelism)(
                joblib.delayed(func)(*args, **kwargs) for _ in range(parallelism)
            )
            return merge_func(results)

        return inner

    if func is None:
        # decorator was used like @parallel(...)
        return decorator
    else:
        # decorator was used like @parallel, without parens
        return decorator(func)


# Threading Version
def parallel_mt(
    func=None, args=(), merge_func=lambda x: x, parallelism=multiprocessing.cpu_count()
):
    def decorator(func: Callable):
        functools.wraps(func)

        def inner(*args, **kwargs):
            pool = multiprocessing.pool.ThreadPool(parallelism)
            results = [pool.apply_async(func, args, kwargs) for _ in range(parallelism)]
            results = [r.get() for r in results]
            pool.close()
            pool.join()
            return merge_func(results)

        return inner

    if func is None:
        # decorator was used like @parallel(...)
        return decorator
    else:
        # decorator was used like @parallel, without parens
        return decorator(func)


@test_decorator
def test_multiprocessing():
    @timer
    @parallel_mp(merge_func=lambda li: sorted(set(itertools.chain(*li))))
    def generate_primes(
        domain: int = 1000 * 1000, num_attempts: int = 1000
    ) -> list[int]:
        primes: set[int] = set()
        random.seed(time.time())
        for _ in range(num_attempts):
            candidate: int = random.randint(4, domain)
            if sympy.isprime(candidate):
                primes.add(candidate)
        return sorted(primes)

    print(len(generate_primes()))


@test_decorator
def test_multithreading():
    @timer
    @parallel_mt(merge_func=lambda li: sorted(set(itertools.chain(*li))))
    def generate_primes(
        domain: int = 1000 * 1000, num_attempts: int = 1000
    ) -> list[int]:
        primes: set[int] = set()
        random.seed(time.time())
        for _ in range(num_attempts):
            candidate: int = random.randint(4, domain)
            if sympy.isprime(candidate):
                primes.add(candidate)
        return sorted(primes)

    print(len(generate_primes()))


# Stack Trace Execution of the code
def stacktrace(func=None, exclude_files=["anaconda"]):
    def tracer_func(frame, event, arg):
        co = frame.f_code
        func_name = co.co_name
        caller_filename = frame.f_back.f_code.co_filename
        if func_name == "write":
            return  # ignore write() calls from print statements
        for file in exclude_files:
            if file in caller_filename:
                return  # ignore in ipython notebooks
        args = str(tuple([frame.f_locals[arg] for arg in frame.f_code.co_varnames]))
        if args.endswith(",)"):
            args = args[:-2] + ")"
        if event == "call":
            print(f"--> Executing: {func_name}{args}")
            return tracer_func
        elif event == "return":
            print(f"--> Returning: {func_name}{args} -> {repr(arg)}")
        return

    def decorator(func: Callable):
        def inner(*args, **kwargs):
            sys.settrace(tracer_func)
            func(*args, **kwargs)
            sys.settrace(None)

        return inner

    if func is None:
        # decorator was used like @stacktrace(...)
        return decorator
    else:
        # decorator was used like @stacktrace, without parens
        return decorator(func)


@test_decorator
def test_stacktrace():
    def b():
        print("...")

    @stacktrace
    def a(arg):
        print(arg)
        b()
        return "world"

    print(a("foo"))


# Stack Trace Execution for a Class
def traceclass(cls: type):
    def make_traced(cls: type, method_name: str, method: Callable):
        def traced_method(*args, **kwargs):
            print(f"--> Executing: {cls.__name__}::{method_name}()")
            return method(*args, **kwargs)

        return traced_method

    for name in cls.__dict__.keys():
        if callable(getattr(cls, name)) and name != "__class__":
            setattr(cls, name, make_traced(cls, name, getattr(cls, name)))
    return cls


@test_decorator
def test_traceclass():
    @traceclass
    class Foo:
        i: int = 0

        def __init__(self, i: int = 0):
            self.i = i

        def increment(self):
            self.i += 1

        def __str__(self):
            return f"This is a {self.__class__.__name__} object with i = {self.i}"

    f1 = Foo()
    f2 = Foo(4)
    f1.increment()
    print(f1)
    print(f2)


# Singleton
def singleton(cls: type):
    def __new__singleton(cls: type, *args, **kwargs):
        if not hasattr(cls, "__singleton"):
            cls.__singleton = object.__new__(cls)  # type: ignore
        return cls.__singleton  # type: ignore

    cls.__new__ = __new__singleton  # type: ignore
    return cls


@test_decorator
def test_singleton():
    @singleton
    class Foo:
        i: int = 0

        def __init__(self, i: int = 0):
            self.i = i

        def increment(self):
            self.i += 1

        def __str__(self):
            return f"This is a {self.__class__.__name__} object with i = {self.i}"

    @singleton
    class Bar:
        i: int = 0

        def __init__(self, i: int = 0):
            self.i = i

        def increment(self):
            self.i += 1

        def __str__(self):
            return f"This is a {self.__class__.__name__} object with i = {self.i}"

    f1 = Foo()
    f2 = Foo(4)
    f1.increment()
    b1 = Bar(9)
    print(f1)
    print(f2)
    print(b1)
    print(f1 is f2)
    print(f1 is b1)


def run_tests():
    test_repeat()
    test_class_decorator()
    test_validate_inputs()
    try:
        test_retry()
    except Exception:
        pass
    test_deprecated()
    test_multithreading()
    test_multiprocessing()
    test_stacktrace()
    test_traceclass()
    test_singleton()


if __name__ == "__main__":
    run_tests()
    # test_interval_timer()
