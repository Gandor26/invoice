import torch as tc
from torch.utils.data import Sampler

class SubsetWeightedSampler(Sampler):
    def __init__(self, indices, weights, num_samples):
        assert len(indices) == len(weights)
        self.indices = indices
        self.weights = tc.as_tensor(weights, dtype=tc.double)
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __iter__(self):
        idx = tc.multinomial(self.weights, self.num_samples, replacement=True)
        return (self.indices[i] for i in idx)


class SubsetSequentialSampler(Sampler):
    def __init__(self, indices):
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __iter__(self):
        return iter(self.indices)
