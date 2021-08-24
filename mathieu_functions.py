# Author: Leonardo Flores Torres

import numpy as np
import numpy.linalg
from scipy import integrate


# Initial values
M = 1.0
G = 1.0
L = 1.0
H = 0.06

U0 = M * G * L
Q = 4 * M * L ** 2 * U0 / H ** 2


###############################################
# Matrixes
###############################################


# a0, a2, a4, ...
def matrix_a_even(n: int):

    diag = [(2 * i) ** 2 for i in range(n)]
    matrix = np.zeros((n, n))

    for i in range(n):
        matrix[i, i] = diag[i]

        if i + 1 < n:
            matrix[i, i + 1] = Q

        if i > 0:
            matrix[i, i - 1] = Q

    matrix[0, 1] = np.sqrt(2) * Q
    matrix[1, 0] = np.sqrt(2) * Q

    # Unordered lists of eigen values and eigen vectors
    vals, vects = np.linalg.eig(matrix)

    # To change column eigen vectors to rows
    vects = np.transpose(vects)

    # Sorting the eigen values to match the ascending order
    # convention, with their respective eigen vectors
    indx = np.argsort(vals)
    vals, vects = vals[indx], vects[indx]

    vects[:, 0:1] = vects[:, 0:1] / np.sqrt(2)

    return vals, vects


# b2, b4, b6, ...
def matrix_b_even(n: int):
    diag = [(2 * i) ** 2 for i in range(1, n + 1)]
    matrix = np.zeros((n, n))

    for i in range(n):
        matrix[i, i] = diag[i]

        if i + 1 < n:
            matrix[i, i + 1] = Q

        if i > 0:
            matrix[i, i - 1] = Q

    # Unordered lists of eigen values and eigen vectors
    vals, vects = np.linalg.eig(matrix)

    # To change column eigen vectors to rows
    vects = np.transpose(vects)

    # Sorting the eigen values to match the ascending order
    # convention, with their respective eigen vectors
    indx = np.argsort(vals)

    return vals[indx], vects[indx]


# a1, a3, a5, ...
def matrix_a_odd(n: int):

    diag = [(1 + 2 * i) ** 2 for i in range(n)]
    matrix = np.zeros((n, n))

    for i in range(n):
        matrix[i, i] = diag[i]

        if i + 1 < n:
            matrix[i, i + 1] = Q

        if i > 0:
            matrix[i, i - 1] = Q

    matrix[0, 0] = 1 + Q

    # Unordered lists of eigen values and eigen vectors
    vals, vects = np.linalg.eig(matrix)

    # To change column eigen vectors to rows
    vects = np.transpose(vects)

    # Sorting the eigen values to match the ascending order
    # convention, with their respective eigen vectors
    indx = np.argsort(vals)

    return vals[indx], vects[indx]


# b1, b3, b5, ...
def matrix_b_odd(n: int):

    diag = [(1 + 2 * i) ** 2 for i in range(n)]
    matrix = np.zeros((n, n))

    for i in range(n):
        matrix[i, i] = diag[i]

        if i + 1 < n:
            matrix[i, i + 1] = Q

        if i > 0:
            matrix[i, i - 1] = Q

    matrix[0, 0] = 1 - Q

    # Unordered lists of eigen values and eigen vectors
    vals, vects = np.linalg.eig(matrix)

    # To change column eigen vectors to rows
    vects = np.transpose(vects)

    # Sorting the eigen values to match the ascending order
    # convention, with their respective eigen vectors
    indx = np.argsort(vals)

    return vals[indx], vects[indx]


###############################################
# Infinite series
###############################################


# Even Mathieu function of period Pi
def ce_even(x: list, vect: list):
    sum = lambda i: np.sum(
        [element * np.cos(indx * (i - np.pi)) for indx, element in enumerate(vect)]
    )
    return np.array([sum(i) for i in x])


def ce_even_v2(x: float, vect: list):
    sum = [element * np.cos(indx * (x - np.pi)) for indx, element in enumerate(vect)]

    return np.sum(sum)


# Odd Mathieu function of period Pi
def se_even(x: list, vect: list):
    sum = lambda i: np.sum(
        [element * np.sin(indx * (i - np.pi)) for indx, element in enumerate(vect)]
    )
    return np.array([sum(i) for i in x])


def se_even_v2(x: float, vect: list):
    sum = [element * np.sin(indx * (x - np.pi)) for indx, element in enumerate(vect)]

    return np.sum(sum)

###############################################
# State related functions
###############################################


def mathieu_fourier(n: int, order: int):
    # The order is used as to increase the accuracy of the characteristic
    # values and the Fourier coefficients of the Mathieu functions
    # Ce and Se

    eig_vals_a, eig_vects_a = matrix_a_even(order)
    eig_vals_b, eig_vects_b = matrix_b_even(order)

    eig_vals = np.concatenate((eig_vals_a, eig_vals_b))
    eig_vects = np.concatenate((eig_vects_a, eig_vects_b))

    indx = np.argsort(eig_vals)
    char_vals = eig_vals[indx]
    fourier_coeff = eig_vects[indx]

    # Still do not know if I should restrain the ammount of returned
    # Fourier coefficients!
    return char_vals[:n], fourier_coeff[:n]


def energy(val: float):
    return (H ** 2 / (8 * M * L ** 2)) * val + U0


def energy_crit(vals: list):
    for indx, val in enumerate(vals):
        if energy(val) > 2 * U0:
            n_crit = indx - 1
            break
    return n_crit, energy(vals[n_crit])


def gauss_coeff(nbar: float, n: int, sigma: float):
    return np.exp(-((n - nbar) ** 2) / (2 * sigma))


# Normalization factor for the eigen functions
def norm(nbar: int, n_max: int, sigma: float):
    sum = np.array([gauss_coeff(nbar, i, sigma) for i in range(n_max)])
    sum = np.sum(sum**2)

    return 1 / np.sqrt(sum)


# Eigen functions
def eigen_state(n: int, x: list, vects: list):
    norm_factor = 1 / np.sqrt(np.pi)

    if n % 2 == 0:
        return ce_even(x, vects[n]) * norm_factor
    else:
        return se_even(x, vects[n]) * norm_factor


def eigen_state_v2(n: int, x: float, vects: list):
    norm_factor = 1 / np.sqrt(np.pi)

    if n % 2 == 0:
        return ce_even_v2(x, vects[n]) * norm_factor
    else:
        return se_even_v2(x, vects[n]) * norm_factor


# Quantum state
def state(nbar: int, x: list, sigma: float, vects: list):
    n_max = len(vects)
    sum = [gauss_coeff(nbar, i, sigma) * eigen_state(i, x, vects) for i in range(n_max)]
    sum = np.transpose(sum)

    return norm(nbar, n_max, sigma) * np.sum(sum, axis=1)


def state_v2(nbar: int, x: float, sigma: float, vects: list):
    n_max = len(vects)
    sum = [gauss_coeff(nbar, i, sigma) * eigen_state_v2(i, x, vects) for i in range(n_max)]

    return norm(nbar, n_max, sigma) * np.sum(sum)
