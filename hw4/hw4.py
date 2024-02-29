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

# plot_dir = 'plots'
# os.makedirs(plot_dir, exist_ok=True)
plot_fname = 'hw4.png'

def markov(x, n):
    y_vals = [0.0] * len(x)
    return y_vals

def chebyshev(x, n):
    return n / (x ** 2)

def plot_results():
    n = 3
    # x_values = np.linspace(0.01, 4, 400)
    # n_values = list(range(1, 6))
    x_vals = np.sum(np.random.uniform(0, 1, (NUM_SIMULATIONS, n)) ** 2, axis=1)
    x_range = np.linspace(0, 4 * n, 50)

    sim_probs = [np.mean(x_vals >= x) for x in x_range]

    plt.figure()
    # for n in n_values:
    #     plt.plot(x_values, chebyshev(x_values * n, n), label=f'Chebyshev n={n}')
    plt.plot(x_range, sim_probs, label="Simulated Probabilities")

    markov_bound = markov(x_range, n=None)
    plt.plot(x_range, markov_bound, label="Markov")

    for label in ['Chebyshev', 'Chernoff', 'Hoeffding']:
        val = np.random.random(1)
        plt.plot(x_range, [val] * len(x_range), label=f"TODO {label}")

    plt.title('Concentration Inequalities')
    plt.xlabel('x')
    plt.ylabel('P(X >= x)')
    plt.legend()
    plt.grid(True)
    plt.show()
    plt.savefig(plot_fname)
def main():
	plot_results()

def main_old():
    if RUN_TRIALS:
        test_buckets = build_test_buckets(X_test, Y_test)
        params = [(m, n, test_buckets) for m in m_values for n in n_values]

        # each config is independent, so can run in a separate thread
        # can plot once all are completed
        with Pool(processes=cpu_count() - 2) as pool:
            results = pool.map(run_config, params)

        # Save the results to a pickle file
        with open(results_pkl, 'wb') as file:
            pickle.dump(results, file)

    plot_results()


if __name__ == "__main__":
    main()
