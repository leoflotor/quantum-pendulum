# Author: Leonardo Flores Torres

# from math import fsum
from mathieu_functions import *
from timeit import default_timer as timer

order = 200
n = 81

# Plot of the characteristic values (with a total of n)
# [a0, b2, a2, b4, ...], [A0, B2, A2, B4, ...]
char_vals, fourier_coeff = mathieu_fourier(n, order)

plt.plot(char_vals, ".")
plt.grid()
plt.show()

# Plot of the energy for the desired number of total energy levels
energy_n = [energy(val) for val in char_vals]

plt.plot(energy_n, ".")
plt.grid()
plt.show()

# Plot of the Ce and Se even functions
x = np.linspace(-np.pi, np.pi, 101)

y_a = ce_even(4, x, fourier_coeff)
y_b = se_even(5, x, fourier_coeff)
state = phi(5, x, fourier_coeff)
#
plt.plot(x, y_a, "--", label="Ce")
plt.plot(x, y_b, "--", label="Se")
plt.plot(x, state, ".", label="Eigen state")
plt.title(f"With a total of {order} Fourier coefficients")
plt.legend()
plt.grid()
plt.show()

# start = timer()
# vals, vects = matrix_a_even_2(100)
# end = timer()
