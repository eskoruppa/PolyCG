import numpy as np
from typing import List, Tuple, Callable, Any, Dict
from .SO3.so3 import euler2rotmat, rotmat2euler

from .bch import BakerCampbellHaussdorff, Lambda_minus, Lambda_plus


class SO3Composites:
    def __init__(self, commutator_order=6, simplify=True, print_status=False):
        self.bch = BakerCampbellHaussdorff()

        # self.Lambda1 = Lambda_minus
        # self.Lambda2 = Lambda_plus

        self.Lambda1, self.Lambda2 = self.bch.linear_rotation_map(
            commutator_order, simplify=simplify, print_status=print_status
        )

    def single_composite_transform_matrix(
        self, phis: np.ndarray, k: int, l: int, sub_id=-1
    ):
        assert k <= l, "k needs to be lesser or equal to k"
        assert -1 <= sub_id < len(phis) // 3, "invalid sub_id"
        if sub_id == -1:
            sub_id = l

        # print(f'{k} -> {l}')

        mat = np.zeros((len(phis), len(phis)))
        for n in range(k, l + 1):
            # print(f' n = {n}')
            mat[
                sub_id * 3 : (sub_id + 1) * 3, n * 3 : (n + 1) * 3
            ] = self.composite_component_transform(phis, k, l, n)
        return mat

    def composite_component_transform(
        self, phis: np.ndarray, k: int, l: int, n: int
    ) -> np.ndarray:
        lam = np.eye(3)
        for i in range(l - 1, n - 1, -1):
            # print(f'  i = {i}')
            # print(f'   {k},{i},{i+1}')
            lam = np.matmul(lam, self.lambda_minus(phis, k, i, i + 1))
        lamp = self.lambda_plus(phis, k, n - 1, n)
        # print(lamp)
        lam = np.matmul(lam, self.lambda_plus(phis, k, n - 1, n))
        return lam

    def lambda_minus(self, phis: np.ndarray, i1: int, i2: int, j: int) -> np.ndarray:
        assert i1 <= i2, f"i1 should not be larger than i2. Given i1={i1}, i2={i2}"
        assert i2 < j, f"i2 need to be smaller than j. Given i2={i2}, j={j}"
        if i1 == i2:
            phi1 = phis[i1 * 3 : (i1 + 1) * 3]
        else:
            phi1 = self.composite_phi(phis, i1, i2)
        return self.Lambda1(*phi1, *phis[j * 3 : (j + 1) * 3])

    def lambda_plus(self, phis: np.ndarray, i1: int, i2: int, j: int) -> np.ndarray:
        if i2 < i1:
            return np.eye(3)
        if i1 == i2:
            phi1 = phis[i1 * 3 : (i1 + 1) * 3]
        else:
            phi1 = self.composite_phi(phis, i1, i2)
        return self.Lambda2(*phi1, *phis[j * 3 : (j + 1) * 3])

    def composite_phi(self, phis: np.ndarray, i: int, j: int) -> np.ndarray:
        """
        Calculate the composite phi from step i to step j.
        Returns the zero vector if i=j and if j<i it returns
        the phi from j to i in the opposite direction. ($-\phi_{j,i}$)
        """
        if i > j:
            raise ValueError("Forward rotation expected: i >= j is not satisfied")
        mat = np.eye(3)
        for n in range(i, j + 1):
            phi = phis[n * 3 : (n + 1) * 3]
            mat = np.matmul(mat, euler2rotmat(phi))
        phi_ij = rotmat2euler(mat)
        return phi_ij


def composite_phi(phis: np.ndarray, i: int, j: int):
    """
    Calculate the composite phi from step i to step j.
    Returns the zero vector if i=j and if j<i it returns
    the phi from j to i in the opposite direction. ($-\phi_{j,i}$)
    """
    if i > j:
        raise ValueError("Forward rotation expected: i >= j is not satisfied")
    mat = np.eye(3)
    for n in range(i, j + 1):
        phi = phis[n * 3 : (n + 1) * 3]
        mat = np.matmul(mat, euler2rotmat(phi))
    phi_ij = rotmat2euler(mat)
    return phi_ij


def hat_map(x: np.ndarray) -> np.ndarray:
    """
    Maps rotation vectors (Euler vectors) onto the corresponding elements of so(3).
    """
    X = np.zeros((3, 3))
    X[0, 1] = -x[2]
    X[1, 0] = x[2]
    X[0, 2] = x[1]
    X[2, 0] = -x[1]
    X[1, 2] = -x[0]
    X[2, 1] = x[0]
    return X


def vec_map(X: np.ndarray) -> np.ndarray:
    """
    Inverse of the hat map. Maps elements of so(3) onto the corresponding Euler vectors.
    """
    return np.array([X[2, 1], X[0, 2], X[1, 0]])


class CoarseGrain:
    def __init__(self, commutator_order=8, simplify=True, print_status=False):
        self.bch = BakerCampbellHaussdorff()
        self.Lambda1, self.Lambda2 = self.bch.linear_rotation_map(
            commutator_order, simplify=simplify, print_status=print_status
        )
