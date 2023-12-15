import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

from ...tests.ztest import power_function_z_onesided
# Set the ggplot style
plt.style.use('ggplot')


def plot_power(theta_true: float, theta_0: float, n: int,
               sigma: float, alpha: float) -> None:
    """
    Plot the power function for both one-sided and two-sided z-tests.

    Parameters:
    - theta_true (float): True parameter value.
    - theta_0 (float): Hypothesized parameter value under the null hypothesis.
    - n (int): Sample size.
    - sigma (float): Standard deviation of the population.
    - alpha (float): Significance level.

    Returns:
    None
    """
    fig, axes = plt.subplots(2, 1, figsize=(8, 6))

    z_crit = theta_0 + sigma*stats.norm.ppf(1-alpha)/np.sqrt(n)

    # Set up the x-axis values
    left_limit = theta_0 - 6 * sigma/np.sqrt(n)
    right_limit = theta_0 + 6 * sigma/np.sqrt(n)
    x = np.linspace(left_limit, right_limit, 1000)

    # Null hypothesis PDF
    null_hypothesis_pdf = stats.norm.pdf(
        x,
        loc=theta_0, scale=sigma/np.sqrt(n)
        )

    # True Distribution PDF
    alternative_hypothesis_pdf = stats.norm.pdf(
        x,
        loc=theta_true,
        scale=sigma/np.sqrt(n)
        )

    # One-sided test:
    axes[0].plot(x, null_hypothesis_pdf, label='Null Hypothesis PDF')
    axes[0].plot(x, alternative_hypothesis_pdf, label='True Distribution PDF')

    power_region = x[x >= z_crit]
    axes[0].fill_between(
        power_region,
        stats.norm.pdf(power_region, loc=theta_true, scale=sigma/np.sqrt(n)),
        label=r'Power (1-$\beta$)',
        color='orange'
        )

    axes[0].axvline(x=z_crit, label='Rejection Threshold', color='green')
    axes[0].set_title('One-sided')
    axes[0].legend()

    power = 1 - stats.norm.cdf(z_crit, loc=theta_true, scale=sigma/np.sqrt(n))
    print(f'Power one-sided: {power}')

    # Two-sided Test:
    z_crit = theta_0 + sigma*stats.norm.ppf(1-alpha/2)/np.sqrt(n)
    axes[1].plot(x, null_hypothesis_pdf, label='Null Hypothesis PDF')
    axes[1].plot(x, alternative_hypothesis_pdf, label='True Distribution PDF')

    power_region = x[x >= z_crit]
    power = 1 - stats.norm.cdf(z_crit, loc=theta_true, scale=sigma/np.sqrt(n))
    axes[1].fill_between(
        power_region,
        stats.norm.pdf(power_region, loc=theta_true, scale=sigma/np.sqrt(n)),
        label=r'Power (1-$\beta$)',
        color='orange'
        )

    power_region = x[x <= -z_crit + 2*theta_0]
    power += stats.norm.cdf(
        -z_crit + 2*theta_0,
        loc=theta_true,
        scale=sigma/np.sqrt(n)
        )
    print(f'Power two-sided: {power}')

    axes[1].fill_between(
        power_region,
        stats.norm.pdf(power_region, loc=theta_true, scale=sigma/np.sqrt(n)),
        color='orange'
        )
    axes[1].axvline(x=z_crit, label='Rejection Threshold', color='green')
    axes[1].axvline(x=-z_crit + 2*theta_0, color='green')
    axes[1].set_title('Two-sided')

    axes[0].set_ylabel('Probability Density')
    axes[1].set_xlabel('Sample Mean')
    axes[1].set_ylabel('Probability Density')
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_power_function(theta_0: float, theta_true: float,
                        n: int, sigma: float, alpha: float) -> None:
    """
    Plot the power function for a one-sided z-test.

    Parameters:
    - theta_0 (float): Hypothesized parameter value under the null hypothesis.
    - theta_true (float): True parameter value.
    - n (int): Sample size.
    - sigma (float): Standard deviation of the population.
    - alpha (float): Significance level.

    Returns:
    None
    """
    theta = np.linspace(90, 120, 1000)
    power = power_function_z_onesided(theta, theta_0, sigma, n, alpha)
    power_n32 = power_function_z_onesided(theta, theta_0, sigma, 2*n, alpha)
    power_n64 = power_function_z_onesided(theta, theta_0, sigma, 4*n, alpha)

    plt.figure(figsize=(7, 5))
    plt.plot(theta, power, label='Power function, n=16')
    plt.plot(theta, power_n32, linestyle='--', label='Power function, n=32')
    plt.plot(theta, power_n64, linestyle='--', label='Power function, n=64')
    plt.axvline(x=theta_true, linestyle=':',
                color='black', label=r'True $\theta$')
    plt.axvline(x=theta_0, linestyle='--',
                color='black', label=r'$\theta_0$')
    plt.ylabel('Power')
    plt.xlabel(r'$\theta$')
    plt.title('Power functions for one-sided z-test.')
    plt.legend()
    plt.show()
