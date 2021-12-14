"""Return the prediction at a set of points."""
from partition import Partition
from generatedata import GenerateData
from partitiondata import Data
import numpy as np
from linearprogram import LinearProgram


class Query:
    def __init__(
        self,
        L,
        epsilon,
        distribution_x=("beta", [1]),
        distribution_y=("beta", [1]),
        seed=42,
        sample_size=100,
        query_size=5,
    ):
        self.L = L
        self.epsilon = epsilon
        self.distribution_x = distribution_x
        self.distribution_y = distribution_y
        self.seed = seed
        self.partition = Partition(self.L, self.epsilon, self.seed)
        self.partition_endpoints = self.partition.endpoints
        self.sample_size = sample_size
        self.query_size = query_size
        self.datagenerate = GenerateData(
            self.sample_size,
            self.seed,
            distribution_x=self.distribution_x,
            distribution_y=self.distribution_y,
        )
        self.x = self.datagenerate.x
        self.y = self.datagenerate.y
        self.partition_data = Data(self.partition_endpoints, self.x)
        self.partition_locs = self.partition_data.partition_loc
        self.partition_types = self.partition_data.partition_type

    def fit_long_ints(self):
        partition_fits = np.empty(self.sample_size)
        for i, loc in enumerate(self.partition_locs):
            if i % 2 != 0:
                partition_fits[i] = "short"
                pass
            elif i - 1 == self.sample_size:
                break
            else:
                idx_subset = np.where(
                    self.x >= loc & self.x < self.partition_locs[i + 1]
                )

                lp = LinearProgram(self.x[idx_subset], self.y[idx_subset], self.L)
                lp_fit = lp.answer
                partition_fits[i] = lp_fit

        self.partition_fits = partition_fits

    def interpolate(self, lipschitz_fit):
        pass

    def classify_pt(self, x):
        start_idx = np.where(self.partition_endpoints > x)[0][0]
        interval_type = self.partition_types[start_idx]

        if interval_type == "short":
            prev_fits = self.partition_fits[start_idx - 1]
            next_fits = self.partition_fits[start_idx + 1]
        pass

