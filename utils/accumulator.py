
import numpy as np

class Accumulator:

    def __init__(self):
        self._sum = 0.0
        self._count = 0

    def add_values(self, vals):
        self._sum += np.sum(vals)
        self._count += len(vals)

    def add_value(self, val):
        self._sum += val
        self._count += 1

    def get_mean(self):
        if self._count == 0:
            return -1
        return self._sum / self._count

    def get_count(self):
        return self._count

