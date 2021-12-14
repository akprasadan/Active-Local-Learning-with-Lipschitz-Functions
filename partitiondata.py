"""Classify datapoints according to their partition."""

import numpy as np
from numba import jit


class Data:
    """
    Determine where an individual x value belongs in the partition, and classify the subinterval.
    """

    def __init__(self, endpoints, x):
        self.endpoints = endpoints
        self.x = x
        self.size = self.x.shape[0]
        self.partition_loc, self.partition_type = self.point_interval()

    def point_interval(self) -> None:
        starts = np.empty(self.size)
        types = np.empty(self.size)
        for i, element in enumerate(self.x):
            start_idx = np.where(self.endpoints > element)[0][0]
            starts[i] = start_idx
            if start_idx % 2 == 0:
                types[i] = "long"
            else:
                types[i] = "short"
        return starts, types
