import logging
from pathlib import Path
from rich.logging import RichHandler


class InfoLogger:
    def __init__(self, logger_name: str, log_path: Path):
        self._logger_name = logger_name
        self._log_path = log_path
        self._init_logger()

    def _init_logger(self):
        logger = logging.getLogger(self._logger_name).handlers.clear()
        logger = logging.getLogger(self._logger_name)
        logger.setLevel(logging.INFO)
        if not self._log_path is None:
            handler = logging.FileHandler(Path(self._log_path) / f'{self._logger_name}.log')
            handler.setLevel(logging.INFO)
            format = logging.Formatter('%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')
            handler.setFormatter(format)
            logger.addHandler(handler)
        self.logger = logger
        return None

    def log(self, message: str = 'test', prefix: str = None):
        if prefix is None:
            self.logger.info(f'{message}')
        else:
            self.logger.info(f'--{prefix}-- {message}')
        return None


def generate_logger(name,logfile=None,stdout=True,level=logging.INFO):
        """Return a logger that by default logs to stdout with log level info."""
        formatter = logging.Formatter('%(asctime)s|%(name)s|%(levelname)s|%(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        logger = logging.getLogger(name)
        logger.setLevel(level)
        logger.propagate=False
        handlers=[]

        if not logfile is None:
            file_handler = logging.FileHandler(logfile)
            file_handler.setFormatter(formatter)
            file_handler.setLevel(level)
            handlers.append(file_handler)
        
        if stdout:
            stdout_handler=RichHandler(rich_tracebacks=True,show_time=True,show_level=True,show_path=False)
            stdout_handler.setFormatter(logging.Formatter('%(name)s\t%(message)s'))
            stdout_handler.setLevel(level)
            handlers.append(stdout_handler)

        for handler in handlers:
            logger.addHandler(handler)

        return logger


