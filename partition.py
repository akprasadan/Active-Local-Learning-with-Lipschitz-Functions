"""
Partitions the data as in Algorithm 1 of 'Active Local Learning' into long and short intervals.
"""

import numpy as np
from numpy.random import default_rng


class Partition:
    """
    Obtain the random partition, parametrized by the endpoints.
    """

    def __init__(self, L: float, epsilon: float, seed):
        self.L = L
        self.epsilon = epsilon
        self.epsilon_inverse = np.ceil(1 / epsilon)
        self.offset = self.random_offset()
        self.short_interval = (1 / self.L) * self.epsilon_inverse
        self.long_interval = 1 / self.L
        self.endpoints = self.interval_partition()
        self.seed = seed

    def random_offset(self) -> float:
        """Choose the first non-zero element of the partition, b_1."""

        rng = default_rng(seed=self.seed)

        offset = (1 / self.L) * rng.integers(
            low=1, high=self.epsilon_inverse, endpoint=True
        )
        if not 0 < offset < 1:
            raise Exception(f"Random offset (b_1 = {offset}) is too large.")

        return offset

    def endpoint_calculator(self, i) -> float:
        """Calculate the endpoint of the ith subinterval. 
        i = 0 returns the offset, while i = 1 returns 1 offset + 1 short, and so on."""

        short_term = ((i + 1) // 2) * (1 / self.L)
        long_term = (i // 2) * (1 / self.L) * self.epsilon_inverse
        return self.offset + short_term + long_term

    def partition_size(self) -> int:
        """Determine how large the partition generated will be."""

        q = self.epsilon_inverse
        r = (1 - self.offset) * self.L
        approx_number = (r - 1 / 2) * 2 / (1 * (1 + q))

        approx_integer = np.floor(approx_number).astype(int)
        candidates = [approx_integer + i for i in range(-3, 3)]
        candidate_vals = np.array(
            [self.endpoint_calculator(candidate) for candidate in candidates]
        )

        first_index = np.where(candidate_vals > 1)[0][0]

        return int(candidates[first_index] + 1)

    def interval_partition(self) -> np.ndarray:
        """Obtain all endpoints of the partition, having calculated the desired size."""

        endpoints = [0, self.offset]
        partition_length = self.partition_size()

        for i in range(1, partition_length):
            if i % 2 == 0:
                next_endpoint = endpoints[-1] + self.short_interval
            else:
                next_endpoint = endpoints[-1] + self.long_interval

            endpoints.append(next_endpoint)
        if endpoints[-1] > 1:
            endpoints.pop()
            if endpoints[-1] < 1:
                endpoints.append(1)
            return endpoints

        return np.array(endpoints)

