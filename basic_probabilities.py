import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import geom


def proportions_viz():
    # Data
    outcomes = ['For', 'Against']
    proportions = [0.71, 0.29]  # Proportions of 'For' and 'Against' outcomes

    # Create bar chart
    plt.bar(outcomes, proportions, color=['pink', 'purple'])

    # Add labels and title
    plt.xlabel('Outcome')
    plt.ylabel('Proportion')
    plt.title('Proportion of "For" and "Against" Outcomes')

    # Show plot
    plt.show()


def geometric_dist():
    # Parameter
    p = 0.71

    # Generate x values
    x = np.arange(1, 200)

    # Calculate probabilities
    probabilities = geom.pmf(x, p)

    # Find the threshold where probability drops below 0.5%
    threshold_index = np.where(probabilities < 0.005)[0][0]

    # Plot the distribution
    plt.bar(x, probabilities, color='blue')

    # Add labels and title
    plt.xlabel('Number of Meteorites')
    plt.ylabel('Probability')
    plt.title('Geometric Distribution (p=0.71)')

    # Add vertical line for threshold
    plt.axvline(x=threshold_index, color='red', linestyle='--', label='Threshold (0.5%)')

    # Calculate and plot expectation and median
    expectation = geom.mean(p)
    median = geom.median(p)
    plt.axvline(x=expectation, color='green', linestyle='--', label='Expectation')
    plt.axvline(x=median, color='purple', linestyle='--', label='Median')

    # Add legend
    plt.legend()

    # Show plot
    plt.show()


if __name__ == '__main__':
    proportions_viz()
    geometric_dist()
