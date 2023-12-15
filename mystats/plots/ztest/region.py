import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Set the ggplot style
plt.style.use('ggplot')


def plot_rejection_regions(null_mu, std_dev, alpha):
    # Set the ggplot style
    plt.style.use('ggplot')

    x = np.linspace(null_mu - 4*std_dev, null_mu + 4*std_dev, 1000)
    null_pdf = stats.norm.pdf(x, loc=null_mu, scale=std_dev)

    fig, axes = plt.subplots(3, 1, figsize=(10, 6))

    for i, alternative in enumerate(['less', 'greater', 'two-sided']):
        ax = axes[i]

        if alternative == 'less':
            q = alpha
            z_crit = stats.norm.ppf(q, loc=null_mu, scale=std_dev)

            ax.plot(x, null_pdf, label='Null Hypothesis PDF')
            ax.axvline(x=z_crit,  label='Rejection Threshold', color='r')

            alpha_region = x[x <= z_crit]
            ax.fill_between(
                alpha_region,
                0,
                stats.norm.pdf(alpha_region, loc=null_mu, scale=std_dev),
                alpha=0.3,
                label=r'Type I Error Region ($\alpha$)',
                color='r'
                )

        elif alternative == 'greater':
            q = 1 - alpha

            z_crit = stats.norm.ppf(q, loc=null_mu, scale=std_dev)

            ax.plot(x, null_pdf, label='Null Hypothesis PDF')
            ax.axvline(x=z_crit,  label='Rejection Threshold', color='r')

            alpha_region = x[x >= z_crit]
            ax.fill_between(
                alpha_region,
                0,
                stats.norm.pdf(alpha_region, loc=null_mu, scale=std_dev),
                alpha=0.3,
                label=r'Type I Error Region ($\alpha$)',
                color='r'
                )

        else:
            q = alpha/2

            z_crit = stats.norm.ppf(q, loc=null_mu, scale=std_dev)

            ax.plot(x, null_pdf, label='Null Hypothesis PDF')
            ax.axvline(x=z_crit,  label='Rejection Threshold', color='r')
            ax.axvline(x=-z_crit, color='r')
            alpha_region = x[x <= z_crit]
            ax.fill_between(
                alpha_region,
                0,
                stats.norm.pdf(alpha_region, loc=null_mu, scale=std_dev),
                alpha=0.3,
                label=r'Type I Error Region ($\alpha$)',
                color='r'
                )
            alpha_region = x[x >= -z_crit]
            ax.fill_between(
                alpha_region,
                0,
                stats.norm.pdf(alpha_region, loc=null_mu, scale=std_dev),
                alpha=0.3,
                color='r'
                )

        ax.set_title(f'Alternative: {alternative}')
        ax.legend(loc='upper right')
        ax.grid(True)
    plt.tight_layout()
    plt.show()
