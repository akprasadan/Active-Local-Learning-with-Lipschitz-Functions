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
        self.point_interval()
        if self.start_idx % 2 == 0:
            self.interval_type = "long"
        else:
            self.interval_type = "short"

    def point_interval(self) -> None:
        for element in self.x:
            start_idx = np.where(self.endpoints > element)[0][0]
            end_idx = start_idx + 1
            self.start, self.end = (
                self.endpoints[start_idx],
                self.endpoints[start_idx + 1],
            )
