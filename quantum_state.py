# Author: Leonardo Flores Torres

###############################################
# Quantum State
###############################################


import numpy as np
from numba import njit
from const import U0, L, M, H, G
import mathieu_functions as mf
from scipy.integrate import odeint
from scipy.interpolate import UnivariateSpline


def energy(val: float):
    return (H ** 2 / (8 * M * L ** 2)) * val + U0


@njit
def energy_numba(val: float):
    return (H ** 2 / (8 * M * L ** 2)) * val + U0


def energy_crit(vals: list):
    # for indx, val in enumerate(vals):
    #     if energy(val) > 2 * U0:
    #         n_crit = indx - 1
    #         break
    # return n_crit, energy(vals[n_crit])
    n_crit = np.argmax(energy(vals) > 2 * U0) - 1
    return n_crit, energy(vals[n_crit])


# Time evolution operator
def time_opr(val: float, t: float):
    return np.exp(1j * energy(val) * t / H)


# Gaussian coeficients that describe the eigen states distribution
# for creating the state of the quantum pendulum
def gauss_coeff(nbar: int, n: int, sigma: float):
    # return ((-1) ** np.random.randint(2)) * np.exp(-((n - nbar) ** 2) / (2 * sigma))
    return np.exp(-((n - nbar) ** 2) / (2 * sigma ** 2))


@njit
def gauss_coeff_numba(nbar: int, n: int, sigma: float):
    return np.exp(-((n - nbar) ** 2) / (2 * sigma ** 2))


# Normalization factor for the quantum state
def norm(nbar: int, n_max: int, sigma: float):
    # summation = np.zeros(n_max)
    # for i in range(len(summation)):
    #     summation[i] = gauss_coeff(nbar, i, sigma)
    # return 1 / np.sqrt(np.sum(summation ** 2))

    summation = np.array([gauss_coeff(nbar, i, sigma) for i in range(n_max)])

    return 1 / np.sqrt(np.sum(summation ** 2))



@njit
def norm_numba(nbar: int, n_max: int, sigma: float):
    #   sum = 0

    #   for i in range(n_max):
    #       sum += gauss_coeff_numba(nbar, i, sigma) ** 2

    #   return 1 / np.sqrt(sum)

    summation = np.zeros(n_max)

    for i in range(n_max):
        summation[i] = gauss_coeff_numba(nbar, i, sigma)

    return 1 / np.sqrt(np.sum(summation ** 2))


# Eigen states selection function.
def eigen_state(n: int, x: np.ndarray, vects: np.ndarray):
    norm_factor = 1 / np.sqrt(np.pi)

    # Select Ce or Se depending on n.
    if n % 2 == 0:
        # Select a_i vectors
        return mf.ce_even(int(n / 2), x, vects[0::2]) * norm_factor
    else:
        # Select b_i vetors
        return mf.se_even(int((n - 1) / 2), x, vects[1::2]) * norm_factor


@njit
def eigen_state_numba(n: int, x: list, vects: list):
    norm_factor = 1 / np.sqrt(np.pi)

    # Adjust n for ce & se, respectively
    if n % 2 == 0:
        # Select a_i vectors
        return mf.ce_even_numba(int(n / 2), x, vects[0::2]) * norm_factor
    else:
        # Select b_i vetors
        return mf.se_even_numba(int((n - 1) / 2), x, vects[1::2]) * norm_factor


# Quantum state
def state(nbar: int, sigma: float, x: np.ndarray, vects: np.ndarray):
    # n_max = len(vects)
    # summation = [gauss_coeff(nbar, i, sigma) * eigen_state(i, x, vects) for i in range(n_max)]

    # return norm(nbar, n_max, sigma) * np.sum(summation, axis=0)

    n_max = len(vects)
    summation = np.zeros((n_max, len(x)))

    for i in range(n_max):
        summation[i] = gauss_coeff(nbar, i, sigma) * eigen_state(i, x, vects)

    return norm(nbar, n_max, sigma) * np.sum(summation, axis=0)


# Esta version de la funcion de estado va RAPIDO!
# Esta version de la funcion de estado va RAPIDO!
# Esta version de la funcion de estado va RAPIDO!
@njit
def state_numba(nbar, sigma, x, vects):
    n_max = len(vects)

    # 1ST BLOCK
    #     sum = []

    #     for i in range(n_max):
    #         .append(gauss_coeff(nbar, i, sigma) * eigen_state_numba(i, x, vects))

    #     sum = np.transpose(np.array(sum))
    #     return norm(nbar, n_max, sigma) * np.sum(sum, axis=1)

    # 2ND BLOCK
    summation = np.zeros((n_max, len(x)))

    for i in range(n_max):
        summation[i] = gauss_coeff_numba(nbar, i, sigma) * eigen_state_numba(
            i, x, vects
        )

    sum = np.transpose(summation)

    return norm_numba(nbar, n_max, sigma) * np.sum(sum, axis=1)
    # 3RD BLOCK
    #   n_max = len(vects)
    #   sum = [
    #   gauss_coeff(nbar, i, sigma) * eigen_state_numba(i, x, vects)
    #       for i in range(n_max)
    #       ]
    #   sum = np.transpose(sum)

    #   return norm(nbar, n_max, sigma) * np.sum(sum, axis=1)


# Time dependent quantum state.
def state_time(
    nbar: int,
    sigma: float,
    x: np.ndarray,
    time: float,
    vals: np.ndarray,
    vects: np.ndarray
):
    n_max = len(vals)

    summation = [time_opr(val, time) * gauss_coeff(nbar, indx, sigma) * \
                 eigen_state(indx, x, vects) for indx, val in enumerate(vals)]

    return norm(nbar, n_max, sigma) * np.sum(summation, axis=0)


# Generic function for the expected angle to the nth power.
def generic_integrand(
    nbar: int,
    sigma: float,
    x: np.ndarray,
    time: float,
    vals: np.ndarray,
    vects: np.ndarray,
    powr: int
):
    return (x ** powr) * np.abs(state_time(nbar, sigma, x, time, vals, vects)) ** 2


# Generic integration for the generic expectation to the nth power.
def generic_expectation(
    nbar: int,
    sigma: float,
    x: np.ndarray,
    time: float,
    vals: np.ndarray,
    vects: np.ndarray,
    powr: int
):
    spl = UnivariateSpline(x, generic_integrand(nbar, sigma, x, time, vals, vects, powr), s=0)

    return spl.integral(x[0], x[-1])


def uncertainty(
    nbar: int,
    sigma: float,
    x: np.ndarray,
    time: np.ndarray,
    vals: np.ndarray,
    vects: np.ndarray
):
    exp_data = [[generic_expectation(nbar, sigma, x, t, vals, vects, 2),
                 generic_expectation(nbar, sigma, x, t, vals, vects, 1)]
                for t in time]
    exp_data = np.array(exp_data)

    exp_powr_two = exp_data[:, 0]
    exp_powr_one = exp_data[:, 1]

    unc = np.sqrt(exp_powr_two - exp_powr_one ** 2)

    return unc, exp_powr_one


def expected_energy(
    nbar: int,
    sigma: float,
    vals: np.ndarray
):
    summation = [energy(val) * np.abs(gauss_coeff(nbar, indx, sigma)) ** 2 for indx, val in enumerate(vals)]

    return np.sum(summation) * (norm(nbar, len(vals), sigma) ** 2)


def classic_pend_energy(
    init_en: float,
    time: np.ndarray,
):
    # The model function is the differential equation
    # corresponding to this problem
    def model(u, t):
        return (u[1], - (G / L) * np.sin(u[0]))

    # Initial angle in terms of the initial energy.
    init_ang = + np.arccos(1 - init_en / (M * G * L))

    # The initial velocity.
    init_vel = 0.0

    # Initial conditions.
    init_cond = [init_ang, init_vel]

    # Solution per time-step
    solution = odeint(model, init_cond, time)

    return solution[:, 0]


def classic_pend_angle(
    init_ang: float,
    time: np.ndarray,
):
    # The model function is the differential equation
    # corresponding to this problem
    def model(u, t):
        return (u[1], - (G / L) * np.sin(u[0]))

    # The initial velocity.
    init_vel = 0.0

    # Initial conditions.
    init_cond = [init_ang, init_vel]

    # Solution per time-step
    solution = odeint(model, init_cond, time)

    return solution[:, 0]


def classic_pend_vel(
    init_ang: float,
    init_en: float,
    time: np.ndarray,
):
    # The model function is the differential equation
    # corresponding to this problem
    def model(u, t):
        return (u[1], - (G / L) * np.sin(u[0]))

    # The initial velocity.
    init_vel = (1 / L) * np.sqrt(2 * ((init_en / M) - G * L * (1 - np.cos(init_ang))))

    # Initial conditions.
    init_cond = [init_ang, -init_vel]

    # Solution per time-step
    solution = odeint(model, init_cond, time)

    return solution[:, 0]
