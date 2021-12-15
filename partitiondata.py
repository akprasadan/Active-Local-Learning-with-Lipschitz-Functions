"""Classify datapoints generated by generatedata.py according to their partition generated by partition.py."""

import numpy as np
from numba import jit


class Data:
    """
    Determine where a set of x values belongs in the partition, and classify the corresponding subintervals. A set of endpoints specifying the partition is required.
    """

    def __init__(self, endpoints, x):
        self.endpoints = endpoints
        self.x = x
        self.size = self.x.shape[0]
        self.partition_loc, self.partition_type = self.point_interval()

    def point_interval(self) -> None:
        """
        Perform the classification of points by interval location and type."""
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
