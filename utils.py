import os
import logging

def get_dir(path):
    os.makedirs(path, exist_ok=True)
    return path

def get_logger(logpath=None, level=logging.INFO):
    logger = logging.getLogger('datalogger')
    formatter = logging.Formatter('%(asctime)s:\t\t%(message)s')
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    if logpath is not None:
        file_handler = logging.FileHandler(logpath)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger


