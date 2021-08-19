# Author: Leonardo Flores Torres

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

# from numpy.linalg import eig
# from scipy.special import jv

import numpy.linalg
import scipy.special
import scipy.interpolate


# Initial values
M = 1.0
G = 1.0
L = 1.0
H = 0.06

U0 = M * G * L
Q = 4 * M * L**2 * U0 / H**2


# Bessel functions arguments s(x) & t(x)
s = lambda x: np.sqrt(Q) * np.exp(x)
t = lambda x: np.sqrt(Q) * np.exp(-x)


###############################################
# Matrixes
###############################################

def matrix_a_even(n: int):

    diag = [(2*i)**2 for i in range(n)]
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

    return vals[indx], vects[indx]


def matrix_b_even(n):

    diag = [(2*i)**2 for i in range(1, n+1)]
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


def matrix_a_odd(n):

    diag = [(1 + 2*i)**2 for i in range(n)]
    matrix = np.zeros((n, n))

    for i in range(n):
        matrix[i, i] = diag[i]

        if i + 1 < n:
            matrix[i, i + 1] = q

        if i > 0:
            matrix[i, i - 1] = q

    matrix[0, 0] = 1 + q

    # Unordered lists of eigen values and eigen vectors
    vals, vects = np.linalg.eig(matrix)

    # To change column eigen vectors to rows
    vects = np.transpose(vects)

    # Sorting the eigen values to match the ascending order
    # convention, with their respective eigen vectors
    indx = np.argsort(vals)

    return vals[indx], vects[indx]


def matrix_b_odd(n):

    diag = [(1 + 2*i)**2 for i in range(n)]
    matrix = np.zeros((n, n))

    for i in range(n):
        matrix[i, i] = diag[i]

        if i + 1 < n:
            matrix[i, i + 1] = q

        if i > 0:
            matrix[i, i - 1] = q

    matrix[0, 0] = 1 - q

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
