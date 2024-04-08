import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import geom
from scipy.stats import poisson
from scipy.stats import nbinom


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


def poisson_dist():
    # Parameters
    lambda_ = 71

    # Generate x values
    x = np.arange(0, 200)

    # Calculate probabilities
    probabilities = poisson.pmf(x, lambda_)

    # Find the threshold where probability drops below 0.5%
    threshold_index = np.where(probabilities < 0.005)[0][0]

    # Plot the distribution
    plt.bar(x, probabilities, color='blue')

    # Add labels and title
    plt.xlabel('Number of Meteorites')
    plt.ylabel('Probability')
    plt.title('Poisson Distribution (λ=71)')

    # Add vertical line for threshold
    plt.axvline(x=threshold_index, color='red', linestyle='--', label='Threshold (0.5%)')

    # Calculate and plot expectation and median
    expectation = lambda_
    median = poisson.median(lambda_)
    plt.axvline(x=expectation, color='green', linestyle='--', label='Expectation')
    plt.axvline(x=median, color='purple', linestyle='--', label='Median')

    # Add legend
    plt.legend()

    # Show plot
    plt.show()


def poisson_dist_2():
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import poisson

    # Parameters
    lmbda = 71  # expectation

    # Generate probabilities
    x = np.arange(0, 200)
    probs = poisson.pmf(x, lmbda)

    # Find the number of meteorites until the probability remains less than 0.5%
    threshold = 0.005
    threshold_idx = np.where(probs < threshold)[0][0]

    # Plot the probabilities
    plt.figure(figsize=(10, 6))
    plt.plot(x, probs, marker='o', linestyle='-', color='b')
    plt.xlabel('Number of Meteorites')
    plt.ylabel('Probability')
    plt.title('Probability Distribution of Meteorites Falling on Ocean')
    plt.axvline(x=threshold_idx, color='r', linestyle='--', label=f'Probability < {threshold}% for x > {threshold_idx}')
    plt.legend()

    # Calculate and plot expectation and median
    expectation = poisson.mean(lmbda)
    median = poisson.median(lmbda)
    plt.axvline(x=expectation, color='g', linestyle='--', label=f'Expectation: {expectation:.2f}')
    plt.axvline(x=median, color='purple', linestyle='--', label=f'Median: {median}')
    plt.legend()

    plt.grid(True)
    plt.show()


def negative_binomial_dist():
    # Parameters
    k = 71
    p = 0.5  # Assume 𝜉3 is 0.5

    # Generate x values
    x = np.arange(0, 200)

    # Calculate probabilities
    probabilities = nbinom.pmf(x, k, p)

    # Find the threshold where probability drops below 0.5%
    threshold_index = np.where(probabilities < 0.005)[0][0]

    # Plot the distribution
    plt.bar(x, probabilities, color='blue')

    # Add labels and title
    plt.xlabel('Number of Meteorites')
    plt.ylabel('Probability')
    plt.title('Negative Binomial Distribution (k=71, p=0.5)')

    # Add vertical line for threshold
    plt.axvline(x=threshold_index, color='red', linestyle='--', label='Threshold (0.5%)')

    # Calculate and plot expectation and median
    expectation = nbinom.mean(k, p)
    median = nbinom.median(k, p)
    plt.axvline(x=expectation, color='green', linestyle='--', label='Expectation')
    plt.axvline(x=median, color='purple', linestyle='--', label='Median')

    # Add legend
    plt.legend()

    # Show plot
    plt.show()


def poisson_dist_3():
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import poisson

    # Define the expectation for the Poisson distribution
    λ = 71

    # Generate x values for plotting
    x = np.arange(0, 200)

    # Calculate the Poisson probability mass function (PMF) for each x
    pmf = poisson.pmf(x, λ)

    # Calculate cumulative probabilities
    cumulative_prob = np.cumsum(pmf)

    # Find the point where cumulative probability exceeds 0.995 (1 - 0.005)
    threshold_index = np.argmax(cumulative_prob > 0.995)

    # Plot the PMF
    plt.figure(figsize=(10, 6))
    plt.bar(x, pmf, label='Poisson Distribution (λ=71)', color='skyblue')

    # Plot expectation and median
    expectation = λ
    median = λ + 1 / 3 - 0.02  # Using approximation for Poisson distribution
    plt.axvline(x=expectation, color='red', linestyle='--', label='Expectation')
    plt.axvline(x=median, color='green', linestyle='--', label='Median')

    # Highlight probabilities less than 0.5%
    plt.fill_between(x[:threshold_index], pmf[:threshold_index], color='orange', alpha=0.5, label='Probability < 0.5%')

    plt.xlabel('Number of Meteorites')
    plt.ylabel('Probability')
    plt.title('Poisson Distribution of Meteorites Falling on Ocean')
    plt.legend()
    plt.grid(True)
    plt.show()

    print("Expectation (λ):", expectation)
    print("Median:", median)


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
    poisson_dist_3()
    negative_binomial_dist()
    geometric_dist()
