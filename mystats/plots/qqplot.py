import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.stats import rankdata


def qq_plot(data, title=''):
    n = len(data)

    # Compute empirical quantiles
    sorted_data = np.sort(data)

    q = (rankdata(sorted_data) - 0.5) / n


    # Compute theoretical quantiles
    theoretical_quantiles =  stats.norm.ppf(q, loc=np.mean(data), scale=np.std(data, ddof=1))

    plt.figure(figsize=(6, 6))
    # Plot Q-Q plot
    plt.scatter(theoretical_quantiles, sorted_data, edgecolors='k', facecolors='none')


    y1, y2 = np.quantile(sorted_data, [0.25, 0.75])
    x1, x2 = stats.norm.ppf([0.25, 0.75], loc=np.mean(data), scale=np.std(data, ddof=1))

    k = (y2-y1)/(x2-x1)
    b = y1 - k*x1
    plt.plot(theoretical_quantiles, k*theoretical_quantiles+b, c='r')

    plt.title(f'Q-Q Plot {title}', fontsize=10)
    plt.xlabel('Theoretical Quantiles')
    plt.ylabel('Sample Quantiles')
    plt.grid()
    plt.grid(True)