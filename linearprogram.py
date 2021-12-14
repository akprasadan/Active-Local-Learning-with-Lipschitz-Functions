"""
Find the empirical risk minimizing estimate of a Lipschitz function, using a linear program.
"""

from typing import List, Tuple, Optional
import numpy as np
from scipy.optimize import linprog

np.set_printoptions(formatter={"float": lambda x: "{0:0.6f}".format(x)})


class LinearProgram:
    """
    Solve the linear program on an interval to obtain the estimate of the Lipschitz function."""

    def __init__(self, x: np.ndarray, y: np.ndarray, L: float):
        self.x = x
        self.y = y
        if not np.all(x[:-1] <= x[1:]):
            raise Exception("x-values should be given in increasing order")
        self.L = L
        self.sample_size = x.shape[0]
        self.obj_coeff = np.concatenate(
            (np.ones((self.sample_size,)), np.zeros((self.sample_size,)))
        )
        self.constraint_matrix = self.generate_constraint_matrix()
        self.constraint_upper = self.generate_constraint_upper()
        self.bound = self.generate_bounds()
        self.answer = self.optimize()

    def generate_constraint_upper(self) -> np.ndarray:
        bound = np.concatenate(
            (
                np.zeros((2 * self.sample_size,)),
                self.L * np.diff(self.x) - np.diff(self.y),
                self.L * np.diff(self.x) + np.diff(self.y),
            )
        )

        return bound

    def generate_bounds(self) -> List[Tuple[Optional[int], Optional[int]]]:
        bound1 = [(0, None) for i in range(0, self.sample_size)]
        bound2 = [(-self.y[i], 1 - self.y[i]) for i in range(0, self.sample_size)]
        return bound1 + bound2

    def generate_constraint_matrix(self) -> np.ndarray:
        short_zero_matrix = np.zeros((self.sample_size - 1, self.sample_size))
        off_diag = LinearProgram.diff_op(self.sample_size - 1, self.sample_size)
        identity = np.eye(self.sample_size)

        block = np.block(
            [
                [-identity, -identity],
                [-identity, identity],
                [short_zero_matrix, off_diag],
                [short_zero_matrix, -off_diag],
            ]
        )
        return block

    def optimize(self) -> np.ndarray:
        res = linprog(
            c=self.obj_coeff,
            A_ub=self.constraint_matrix,
            b_ub=self.constraint_upper,
            bounds=self.bound,
        )
        return res.x[self.sample_size :] + self.y

    @staticmethod
    def diff_op(rows: int, cols: int) -> np.ndarray:
        # Construct the vector differencing operator
        # Matrix multiplies to a vector of dimension rows + 1, which when applied would output the consecutive differences in vector of length rows
        matrix = -1 * np.eye(N=rows, M=cols, k=0) + np.eye(N=rows, M=cols, k=1)
        return matrix
