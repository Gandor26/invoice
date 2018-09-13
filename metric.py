import numpy as np
import torch as tc
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

class TopKAccuracy(object):
    def __init__(self, top_k):
        self.top_k = top_k

    def __call__(self, preds, labels):
        _, top = preds.topk(self.top_k, dim=1)
        acc = labels.unsqueeze(dim=1).eq(top).any(dim=1, keepdim=False).float().mean()
        return acc
