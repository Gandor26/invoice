import os
import logging

__all__ = ['get_dir', 'get_logger']

LOGPATH = os.path.expanduser('~/workspace/invoice/default.log')

def get_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def get_logger(log_path=LOGPATH, level=logging.INFO, clear=False):
    if clear:
        f = open(log_path, 'w')
        f.close()
    logger = logging.getLogger('datalogger')
    formatter = logging.Formatter('%(asctime)s:\t\t%(message)s')
    if log_path is None:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    else:
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger


