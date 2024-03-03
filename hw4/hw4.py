"""
Matthew Pacey
AI 539 - HW4
Concentration Inequalities
"""

import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.stats import chi2

DEBUG = True
plot_dir = 'plots'
os.makedirs(plot_dir, exist_ok=True)

# various matplot styles; the final is a custom dash pattern (only used if n-range >= 5)
line_styles = ['-', '--', '-.', ':', (0, (3, 5, 1, 5, 1, 5))]

def plot_new():

    for n in [1, 2, 4, 8]:
        plt.figure()

        # Add vertical line at x=n
        plt.axvline(x=n, color='black', linestyle='--', label='x=n')

        x_stop = 4 * n
        x_range = np.linspace(0.0001, x_stop, 50)

        # simulated
        sim_prob = [1 - chi2.cdf(x, df=n) for x in x_range]
        plt.plot(x_range, sim_prob, label=f"Sim Prob", color='blue')

        # Markov bound: n / x
        y_markov = n / x_range
        plt.plot(x_range, y_markov, label=f"Markov", color='red', linestyle='--')

        # Chebyshev bound: n / x^2
        y_cheb = 2 * n / x_range ** 2
        plt.plot(x_range, y_cheb, label=f"Chebyshev", color='green', linestyle='-.')

        # Chernoff bound: e^(-n x^2 /(2n + 2x)
        y_cher = np.exp(-n * x_range ** 2 / (2*x_range + 2 * n))
        plt.plot(x_range, y_cher, label=f"Chernoff", color='purple', linestyle=':')

        # Hoeffding for multiple k values = P(union A) +  P(not A)
        for i, k in enumerate([2, 4, 16]):
            y_hoeff = 2 * np.exp(-2 * (x_range - n) ** 2 / (n * 2 * k **2))  + 2 * n * np.exp(- k ** 2 / 2)
            plt.plot(x_range, y_hoeff, label=f"Hoeffding ({k=})", color='orange', linestyle=line_styles[i])


        plt.title(f'Concentration Inequalities ({n=})')
        plt.xlabel('x')
        plt.ylabel('P(X >= x)')
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.yscale('log')  # Set logarithmic scale for y-axis
        plt.ylim(0, 20)  # Set maximum y-value
        plt.grid(True)
        plt.tight_layout()
        plot_fname = os.path.join(plot_dir, f"n_{n}.png")
        plt.savefig(plot_fname)
        print(f"Plot created: {plot_fname}")
        plt.show()

if __name__ == "__main__":
    plot_new()
