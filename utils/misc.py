import os
import logging

__all__ = ['get_dir', 'get_logger']

LOGPATH = os.path.expanduser('~/workspace/invoice/default.log')

def get_dir(path):
    os.makedirs(path, exist_ok=True)
    return path

def get_logger(logpath=LOGPATH, level=logging.INFO):
    logger = logging.getLogger('datalogger')
    formatter = logging.Formatter('%(asctime)s:\t\t%(message)s')
    if logpath is None:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    else:
        file_handler = logging.FileHandler(logpath, mode='w')
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger


