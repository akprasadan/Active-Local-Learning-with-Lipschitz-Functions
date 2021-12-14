import numpy as np
from numpy.random import default_rng
from typing import List, Tuple, Dict


class GenerateData:
    def __init__(
        self,
        sample_size: int,
        seed: int,
        distribution_x: Tuple[str, List[float]],
        distribution_y: Tuple[str, List[float]],
    ):
        self.sample_size = sample_size
        self.seed = seed
        self.rng = default_rng(seed=seed)
        self.distribution_list: Dict = {"beta": GenerateData.beta_dist}
        self.distribution_x, self.params_x = distribution_x
        self.distribution_y, self.params_y = distribution_y
        self.x = self.generate_x()
        self.y = self.generate_y()

    def generate_x(self) -> np.ndarray:
        pdf = self.distribution_list[self.distribution_x]

        draw = np.sort(pdf(self.rng, self.sample_size, self.params_x))

        return draw

    def generate_y(self) -> np.ndarray:
        pdf = self.distribution_list[self.distribution_y]

        draw = pdf(self.rng, self.sample_size, self.params_y)

        return draw

    @staticmethod
    def beta_dist(
        range_generator, size: int, params: list
    ) -> np.ndarray:  # uniform by default
        a, b = params
        return range_generator.beta(a=a, b=b, size=size)

