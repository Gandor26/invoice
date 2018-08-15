import numpy as np
import time

class BaseMeter(object):
    def add(self, val):
        raise NotImplementedError
    def reset(self):
        raise NotImplementedError
    def read(self):
        raise NotImplementedError

class AverageMeter(BaseMeter):
    def __init__(self, tag=None):
        self.tag = tag
        self.reset()

    def add(self, val, count=1):
        self.sum += val*count
        self.count += count

    def reset(self):
        self.sum = 0
        self.count = 0

    def read(self):
        assert self.count > 0, 'No observation has been added.'
        return self.sum/self.count

class TimeMeter(BaseMeter):
    def __init__(self):
        self.reset()

    def reset(self):
        self.start_point = time.time()

    def read(self):
        return time.time() - self.start_point
