"""
Script to find the integer characteristic values for the Mathieu Differential
Equations by the matrixial method (easily found in the literature). This is
done by proposing the solution of the differential equation as a Fourier series
expansion; the Fourier series coefficients are also computed in this
script.


Author: Leonardo Flores Torres
"""


import numpy as np
from const import Q


# Matrix for a0, a2, a4, ...
def matrix_a_even(order: int, n: int):
    """
    Computes both even-indexed Mathieu characteristic values and Fourier
    coefficients vectors for the even solution.

    Parameters
    ----------
    order : int
        Related to the order of the matrix, the precision of the characteristic
        values and fourier coefficients increases as the order is higher.
    n : int
        Number of desired characteristic values and fourier coefficients.

    Returns
    -------
    vals : ndarray(dtype=float, ndim=1)
        Characteristic values.
    vects : ndarray(dtype=float, ndim=2)
        Fourier coefficients.
    """

    diag = [(2 * i) ** 2 for i in range(order)]
    matrix = np.zeros((order, order), dtype=np.float64)

    for i in range(order):
        matrix[i, i] = diag[i]

        if i + 1 < order:
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

    return vals[:n], vects[:n]


# Matrix for a1, a3, a5, ...
def matrix_a_odd(order: int, n: int):
    """
    Computes both odd-indexed Mathieu characteristic values and Fourier
    coefficients vectors for the even solution.

    Parameters
    ----------
    order : int
        Related to the order of the matrix, the precision of the characteristic
        values and fourier coefficients increases as the order is higher.
    n : int
        Number of desired characteristic values and fourier coefficients.

    Returns
    -------
    vals : ndarray(dtype=float, ndim=1)
        Characteristic values.
    vects : ndarray(dtype=float, ndim=2)
        Fourier coefficients.
    """

    diag = [(1 + 2 * i) ** 2 for i in range(order)]
    matrix = np.zeros((order, order), dtype=np.float64)

    for i in range(order):
        matrix[i, i] = diag[i]

        if i + 1 < order:
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
    vals, vects = vals[indx], vects[indx]

    return vals[:n], vects[:n]


# Matrix for b2, b4, b6, ...
def matrix_b_even(order: int, n: int):
    """
    Computes both even-indexed Mathieu characteristic values and Fourier
    coefficients vectors for the odd solution.

    Parameters
    ----------
    order : int
        Related to the order of the matrix, the precision of the characteristic
        values and fourier coefficients increases as the order is higher.
    n : int
        Number of desired characteristic values and fourier coefficients.

    Returns
    -------
    vals : ndarray(dtype=float, ndim=1)
        Characteristic values.
    vects : ndarray(dtype=float, ndim=2)
        Fourier coefficients.
    """

    diag = [(2 * i) ** 2 for i in range(1, order + 1)]
    matrix = np.zeros((order, order), dtype=np.float64)

    for i in range(order):
        matrix[i, i] = diag[i]

        if i + 1 < order:
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
    vals, vects = vals[indx], vects[indx]

    return vals[:n], vects[:n]


# Matrix for b1, b3, b5, ...
def matrix_b_odd(order: int, n: int):
    """
    Computes both odd-indexed Mathieu characteristic values and Fourier
    coefficients vectors for the odd solution.

    Parameters
    ----------
    order : int
        Related to the order of the matrix, the precision of the characteristic
        values and fourier coefficients increases as the order is higher.
    n : int
        Number of desired characteristic values and fourier coefficients.

    Returns
    -------
    vals : ndarray(dtype=float, ndim=1)
        Characteristic values.
    vects : ndarray(dtype=float, ndim=2)
        Fourier coefficients.
    """

    diag = [(1 + 2 * i) ** 2 for i in range(order)]
    matrix = np.zeros((order, order), dtype=np.float64)

    for i in range(order):
        matrix[i, i] = diag[i]

        if i + 1 < order:
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
    vals, vects = vals[indx], vects[indx]

    return vals[:n], vects[:n]


# The great list of ordered mathieu characteristic values and
# fourier coefficients
def mathieu_fourier(order: int, n: int):
    """
    Full set of Mathieu characteristic values and Fourier coefficients vectors.
    The accuracy of the characteristic values and the Fourier coefficients
    increases as n increases.

    Parameters
    ----------
    order : int
        Number of Mathieu characteristic values and Fourier coefficients
        vectors to be computed, the precision increases as the usr selects a
        higher order.
    n_max : int
        Number of wanted characteristic values and coefficients.

    Returns
    -------
    vals : ndarray(dtype=float, ndim=1)
        Full set of characteristic values.
    vects : ndarray(dtype=float, ndim=2)
        Full set of Fourier coefficients.
    """

    # Getting a little surplus of values and coefficients to ensure that all
    # the wanted values and coefficients are obtained at the end when the
    # arrays are sliced.
    local_n = int((n / 2) + 2)
    eig_vals_a, eig_vects_a = matrix_a_even(order, local_n)
    eig_vals_b, eig_vects_b = matrix_b_even(order, local_n)

    eig_vals = np.concatenate((eig_vals_a, eig_vals_b))
    eig_vects = np.concatenate((eig_vects_a, eig_vects_b))

    indx = np.argsort(eig_vals)
    vals = eig_vals[indx]
    vects = eig_vects[indx]

    # Still do not know if I should cut the fourier coefficients vectors...?
    return vals[:n+1], vects[:n+1]

