
from contextlib import contextmanager
from pathlib import Path
import time
import os
import sys


def store_metadata_fun(file_path: Path):
    user = os.environ.get("USER")
    box = os.uname().nodename
    script = " ".join(sys.argv)
    with open(file_path, "a") as f:
        f.write(f"Lock file created by {user} on {box} using the {script} script.")


class FileLockError(ValueError):
    pass


def acquire_lock(
    file_path: Path,
    poll_interval=0.1,
    timeout: float | None = None,
    store_metadata: bool = False,
) -> Path:
    use_timeout = timeout is not None
    if not use_timeout:
        timeout = 0

    lock_file = Path(f"{file_path}.lock")
    while lock_file.is_file() and (timeout > 0 or not use_timeout):
        time.sleep(poll_interval)
        timeout -= poll_interval

    if use_timeout and timeout <= 0:
        try:
            with open(lock_file, "r") as f:
                contents = f.read()
        except FileNotFoundError:
            contents = ""

        raise FileLockError(
            f"Unable to acquire lock on {file_path}. The job that created the lock might have died. Consider removing the lockfile using rm {lock_file.absolute()}. Contents of lockfile: {contents}"
        )

    create_empty_file(lock_file)
    if store_metadata:
        store_metadata_fun(lock_file)

    return lock_file


def create_empty_file(file_path: Path):
    with open(file_path, "wb"):
        pass


def release_lock(file_path: Path):
    os.remove(file_path)


@contextmanager
def lock_file(
    file: Path | str,
    poll_interval: float = 0.1,
    timeout: float | None = None,
    store_metadata: bool = True,
):
    """
    Lock a file for exclusive usage. If necessary we can implement a lock mechanism with multilpe read acccess.

    This works by creating a lock file (just an empty file whose existence indicates that the file is lock) under file+'.lock', performing the actions (i.e. reading and writing) on the file and deleting the lock file once done. If the lock file already exists it means another process is using it and we wait until that process removes the file. This means for the lock to work properly all users of the file need to be using this mechanism.

    The lock file mechanism is provided in form of context manager (with). The good part about this is that even if we encounter an error in the user code, the lock file will be removed. Deadlocks can still happen if the process is terminated using SIGKILL. Lock files need to be removed manually in such cases.


    Args:
        file: file path of the file to lock
        poll_interval: time in seconds to wait between checking if the lock has been released. defaults to 0.1 secs but can be reduced for small files. The downside of using a very small poll_interval is that it puts strain on the disk.
        timeout: number of seconds to wait before raising FileLockError. If None wait forever
        store_metadata: store username, hostname and script name in the lockfile


    Examples:

        >>> with lock_file('/htaa/akment/data.parquet') as f:
        >>>        pd.read_parquet(f)
                # or any other file operation just make sure to unindent as soon as the file is no longer needed
        or
        >>> file='/htaa/akment/data.parquet'
        >>> with lock_file(file):
        >>>      pd.read_parquet(file)

    """

    if isinstance(file, str):
        file = Path(file)

    lock_file = acquire_lock(
        file,
        poll_interval=poll_interval,
        timeout=timeout,
        store_metadata=store_metadata,
    )

    try:
        yield file
    finally:
        release_lock(lock_file)
