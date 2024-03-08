"""
Matthew Pacey
AI 539 - HW5
Complexity
"""

from math import comb               # for n choose i
import matplotlib.pyplot as plt
import numpy as np

DEBUG = True
plt_fname = 'hw5.png'


def plot():
    # VC-dimension for our function
    d_vc = 2

    n_values = np.arange(0, 4)

    # growth function
    y_growth = 2 ** n_values
    plt.plot(n_values, y_growth, label=r'Growth $\Delta_F(n)$', color='green')

    # (e * n / d_vc) ** d_vc for each n
    y_values = ((np.e * n_values) / d_vc) ** d_vc
    # alt: n * e^2 / 4
    # y_values = np.e ** 2 * n_values / 4
    plt.plot(n_values, y_values, label=r'$(\frac{e \cdot n}{d_{VC}})^{d_{VC}}$', color='blue', linestyle='-.')

    # sum (n choose i )
    bound_1_vals = np.array([sum([comb(n, i) for i in range(d_vc + 1)]) for n in n_values])
    plt.plot(n_values, bound_1_vals, label=r'$\sum_{i=0}^{d_{VC}} \binom{n}{i}$', color='red', linestyle='--')

    # Add a vertical line at n=d_vc
    plt.axvline(x=d_vc, color='black', linestyle=':', label=r'$n=d_{vc}$')

    plt.xlabel('n')
    # plt.ylabel('$(\frac{e \cdot n}{d_{VC}})^{d_{VC}}$')
    plt.title(r'Complexity Bounds')
    plt.legend()
    plt.grid(True)
    plt.savefig(plt_fname)
    plt.show()


if __name__ == "__main__":
    plot()
