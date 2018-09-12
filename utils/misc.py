import os
import logging

__all__ = ['get_dir', 'get_logger']

DATA_LOG = os.path.expanduser('~/workspace/invoice/data.log')
TRAIN_LOG = os.path.expanduser('~/workspace/invoice/train.log')

console_fmt = logging.Formatter('{asctime}: {pathname}:{lineno} -> {message}', '%m/%d/%Y %H:%M:%S', style='{')
file_fmt = logging.Formatter('{asctime}: {name} -> {message}', '%m/%d/%Y %H:%M:%S', style='{')

data_logger = logging.getLogger('data')
data_logger.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.WARNING)
console_handler.setFormatter(console_fmt)
file_handler = logging.FileHandler(DATA_LOG)
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(file_fmt)
data_logger.addHandler(console_handler)
data_logger.addHandler(file_handler)
train_logger = logging.getLogger('train')
train_logger.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.WARNING)
console_handler.setFormatter(console_fmt)
file_handler = logging.FileHandler(TRAIN_LOG)
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(file_fmt)
train_logger.addHandler(console_handler)
train_logger.addHandler(file_handler)

def get_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def get_logger(name, clear=False):
    root = logging.getLogger(name.split('.')[0])
    if not root.hasHandlers():
        raise ValueError('The logger name specified does not contain a valid root logger')
    if clear:
        if root.name == 'data':
            with open(DATA_LOG, 'w'):
                pass
        elif root.name == 'train':
            with open(TRAIN_LOG, 'w'):
                pass
    logger = logging.getLogger(name)
    return logger


