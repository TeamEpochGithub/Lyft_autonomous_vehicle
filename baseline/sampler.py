from random import Random
import numpy as np

class RandomSampler:
    def __init__(self, seed, start, end):
        self.seed = seed
        self.length = end - start
        self.start = start
        
        self.permutation = np.random.RandomState(seed).permutation(np.arange(end))[start:]

    def __iter__(self):
        for idx in self.permutation:
            yield idx

    def __len__(self):
        return self.length
        