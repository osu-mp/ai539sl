"""
Matthew Pacey
AI 539 - HW4
Concentration Inequalities
"""

import os.path
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
import numpy as np
import pickle
from scipy.io import loadmat

DEBUG = True
NUM_SIMULATIONS = 10000
MIN_N = 1
MAX_N = 4
# plot_dir = 'plots'
# os.makedirs(plot_dir, exist_ok=True)
plot_fname = 'hw4.png'

# various matplot styles; the final is a custom dash pattern (only used if n-range >= 5)
line_styles = ['-', '--', '-.', ':', (0, (3, 5, 1, 5, 1, 5))]



def markov(x, n):
    return n / x

def chebyshev(x, n):
    #y_values = n / x **2
    return n / (x ** 2)

def simulate(n, count=NUM_SIMULATIONS):
    """
    Simulate the given number of times:
    sum(X^2_i) for i = 1 to n
        each X_i is normal from 0 to 1
    :param n:
    :param count:
    :return:
    """
    x_s = np.random.uniform(0, 1, (count, n))
    x_s_sq = x_s ** 2
    sums = np.sum(x_s_sq, axis=1)
    if DEBUG:
        print(f"Simulating with {n=}, {count=}")
        print(f"{x_s=}\n{x_s_sq=}\n{sums=}")
    return sums
def plot_results():
    plt.figure()

    # simulated probs
    for n, line_style in zip(range(MIN_N, MAX_N + 1), line_styles):
        x_vals = simulate(n)
        x_range = np.linspace(0, 4 * n, 50)
        sim_prob = ([np.mean(x_vals >= x) for x in x_range])
        plt.plot(x_range, sim_prob, label=f"Sim Probs ({n=})", color='blue', linestyle=line_style)

    # markov bounds
    for n, line_style in zip(range(MIN_N, MAX_N + 1), line_styles):
        x_range = np.linspace(0.000001, 4 * n, 50)
        x_vals = n / x_range
        prob = ([np.mean(x_vals >= x) for x in x_range])
        plt.plot(x_range, prob, label=f"Markov ({n=})", color='red', linestyle=line_style)

    # chebyshev
    for n, line_style in zip(range(MIN_N, MAX_N + 1), line_styles):
        x_range = np.linspace(0.000001, 4 * n, 50)
        x_vals = n / (x_range ** 2)
        prob = ([np.mean(x_vals >= x) for x in x_range])
        plt.plot(x_range, prob, label=f"Chebyshev ({n=})", color='Green', linestyle=line_style)

    # TODO Chernoff
    # TODO Hoeffding (for a few k values)

    # for n, line_style in zip(range(MIN_N, MAX_N + 1), line_styles):
    #     x_vals = simulate(n)
    #     x_range = np.linspace(0, 4 * n, 50)
    #     sim_probs.append([np.mean(x_vals >= x) for x in x_range])
    #
    #     markov_bound = markov(x_range, n=None)
    #     plt.plot(x_range, markov_bound, label=f"Markov ({n=})", color='red', linestyle=line_style)


    # for label in ['Chernoff', 'Hoeffding']:
    #     val = np.random.random(1)
    #     plt.plot(x_range, [val] * len(x_range), label=f"TODO {label}")

    plt.title('Concentration Inequalities')
    plt.xlabel('x')
    plt.ylabel('P(X >= x)')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(plot_fname)
    plt.show()

def main():
	plot_results()

if __name__ == "__main__":
    main()
    # simulate(n=6, count=4)
