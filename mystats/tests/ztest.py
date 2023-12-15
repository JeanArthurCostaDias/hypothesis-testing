from scipy import stats
import numpy as np


def z_test(data: list[float], popmean: float, popvariance: float,
           alternative: str = 'greater',
           use_sample_variance: str = True) -> dict:
    """
    Perform a one-sample z-test.

    Parameters:
    - data (list[float]): A list of numerical values representing
                        the sample data.
    - popmean (float): The hypothesized population mean
                        under the null hypothesis.
    - popvariance (float): The population variance under the null hypothesis.
    - alternative (str): The alternative hypothesis. Options: 'greater',
                        'less', or 'two_sided'. Default is 'greater'.
    - use_sample_variance (bool): If True, use the sample variance;
                        if False, use the specified population variance.
                        Default is True.

    Returns:
    - dict: A dictionary containing the z-score and p-value.

    Raises:
    - KeyError: If the specified alternative is not one of 'greater',
                'less', or 'two_sided'.
    """

    x_mean = np.mean(data)
    sample_size = len(data)

    if use_sample_variance:
        var = np.var(data, ddof=1)
    else:
        var = popvariance

    z_score = np.sqrt(sample_size)*(x_mean - popmean)/np.sqrt(var)

    if alternative == 'greater':
        p_value = 1 - stats.norm.cdf(z_score)
    elif alternative == 'less':
        p_value = stats.norm.cdf(z_score)
    elif alternative == 'two_sided':
        p_value = 2*stats.norm.cdf(z_score)
    else:
        raise KeyError(f"{alternative} is not implemented. "
                       f"Use: 'greater', 'less' or 'two_sided'")
    return {'z_score': z_score, 'p_value': p_value}


def power_function_z_onesided(theta: float, theta_0: float,
                              sigma: float, n: int,
                              alpha: float = 0.05) -> float:
    """
    Calculate the power of a one-sided z-test.

    Parameters:
    - theta (float): True parameter value.
    - theta_0 (float): Hypothesized parameter value under the null hypothesis.
    - sigma (float): Standard deviation of the population.
    - n (int): Sample size.
    - alpha (float, optional): Significance level, default is 0.05.

    Returns:
    float: The power of the one-sided z-test.

    Example:
    >>> power_function_z_onesided(105, 100, 16, 30)
    0.5266213245617535
    """

    z_crit = stats.norm.ppf(1 - alpha)
    power = 1 - stats.norm.cdf(z_crit + np.sqrt(n) * (theta_0 - theta) / sigma)
    return power
