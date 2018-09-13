import os
import logging

__all__ = ['get_dir', 'get_logger']

CONSOLE_FORMAT = logging.Formatter('{asctime}: {pathname}:{lineno} -> {message}', '%m/%d/%Y %H:%M:%S', style='{')
FILE_FORMAT = logging.Formatter('{asctime}: {name} -> {message}', '%m/%d/%Y %H:%M:%S', style='{')

def _get_log_path(name):
    return os.path.expanduser('~/workspace/invoice/logs/{}.log'.format(name))

def _make_root_logger(name, mode='a'):
    root_logger = logging.getLogger(name)
    root_logger.setLevel(logging.DEBUG)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARNING)
    console_handler.setFormatter(CONSOLE_FORMAT)
    file_handler = logging.FileHandler(_get_log_path(name), mode)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(FILE_FORMAT)
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

_make_root_logger('utils')
_make_root_logger('train')
_make_root_logger('test', mode='w')

def get_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def get_logger(name, clear=False):
    root = name.split('.')[0]
    root_logger = logging.getLogger(root)
    if not root_logger.hasHandlers():
        raise ValueError('The logger name specified does not contain a valid root logger')
    if clear:
        with open(_get_log_path(root), 'w'):
            pass
    logger = logging.getLogger(name)
    return logger


