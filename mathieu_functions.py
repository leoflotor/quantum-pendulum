# Author: Leonardo Flores Torres

###############################################
# Infinite series
###############################################


import numpy as np
from typing import Union
from numba import njit


# Even Mathieu function of period Pi
def ce_even_sum(n: int, x: Union[float, np.ndarray], vects: Union[list, np.ndarray]):
    vect = vects[n]
    summation = [elem * np.cos(indx * (x - np.pi)) for indx, elem in enumerate(vect)]

    return np.sum(summation, axis=0)


@njit
def ce_even_sum_numba(n: int, x: float, vects: list) -> float:
    vect = vects[n]
    sum = 0.0

    for indx in range(len(vect)):
        sum += vect[indx] * np.cos(indx * (x - np.pi))

    return sum


def ce_even(n: int, x: Union[float, np.ndarray], vects: Union[list, np.ndarray]):
    factor = 1

    if np.sign(ce_even_sum(n, 0.0, vects)) != (-1) ** n:
        factor = -1

    return factor * np.array(ce_even_sum(n, x, vects))


@njit
def ce_even_numba(n: int, x: np.ndarray, vects: Union[list, np.ndarray]):
    # if n % 2 == 0:
    #     factor_EO = 1
    # else:
    #     factor_EO = 1  # -1

    if np.sign(ce_even_sum_numba(n, 0.0, vects)) != (-1) ** n:
        factor = -1
    else:
        factor = 1

    summation = np.zeros(len(x))
    for indx in range(len(x)):
        summation[indx] = ce_even_sum_numba(n, x[indx], vects)

    return factor * summation


# Odd Mathieu function of period Pi
def se_even_sum(n: int, x: Union[float, np.ndarray], vects: Union[list, np.ndarray]):
    vect = vects[n]
    summation = [elem * np.sin((indx + 1) * (x - np.pi)) for indx, elem in enumerate(vect)]

    return np.sum(summation, axis=0)


@njit
def se_even_sum_numba(n: int, x: float, vects: list) -> float:
    vect = vects[n]
    sum = 0.0

    for indx in range(len(vect)):
        sum += vect[indx] * np.sin((indx + 1) * (x - np.pi))

    return sum


def se_even(n: int, x: Union[float, np.ndarray], vects: Union[list, np.ndarray]) -> np.ndarray:
    # This method does not allow to run the function with individual floats!
    # x1_indx = np.nonzero(x > 0.0)[0][0]
    # x1 = x[x1_indx]

    x1 = 1e-5  # WARNING!!!
    x0 = 0.0

    y1 = se_even_sum(n, x1, vects)
    y0 = se_even_sum(n, x0, vects)

    factor = -1

    # Sign of the slope.
    if np.sign(y1 - y0) != (-1) ** (n + 1):
        factor = 1

    return factor * se_even_sum(n, x, vects)


@njit
def se_even_numba(n: int, x: list, vects: list) -> np.ndarray:
    how_close = 1e-5  # WARNIGN!!!

    if n % 2 == 0:
        factor_EO = 1
    else:
        factor_EO = 1  # -1

    if se_even_sum_numba(n, -np.pi + how_close, vects) < 0:
        factor = -1
    else:
        factor = 1

    summation_ith = np.zeros(len(x))
    for indx in range(len(x)):
        summation_ith[indx] = se_even_sum_numba(n, x[indx], vects)

    return factor_EO * factor * summation_ith
