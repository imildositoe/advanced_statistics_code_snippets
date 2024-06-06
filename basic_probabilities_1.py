import numpy as np
from scipy.stats import poisson
import matplotlib.pyplot as plt

lambda_value = 71
threshold_value = 0.005

# First we generate the K values from 1 to 100 to be used to compute the probabilities until we find the range
# of K values where the probability remains less than 0.5% (0.005) for any bigger number of meteorites
values_of_k = np.arange(1, 100)
probs = poisson.pmf(values_of_k, lambda_value)

# Determine the range of k where probabilities are above 0.5%
valid_k_values = values_of_k[probs >= threshold_value]

# Here we calculate the expectation value by attributing the given lambda value in the task
# Then we compute the approximation value of median based on Wikipedia-Poisson formula
expectation_value = lambda_value
median_value = round(lambda_value + (1 / 3) - (1 / (50 * lambda_value)), 1)

# Here we print in the console the calculated Expectation and Median values
print("Expectation value: ", expectation_value)
print("Median value: ", median_value)

# Here we plot the graph displaying the probabilities of meteorites falling on an ocean within the given conditions
# We also display in the same graph the representation lines of the Expectation and the Median
plt.figure(figsize=(10, 5))
plt.bar(valid_k_values, probs[probs >= threshold_value], color='c', alpha=0.7, label='P(X = k)')
plt.axvline(x=lambda_value, color='k', linestyle='--', label=f'Expectation: {lambda_value}')
plt.axvline(x=median_value, color='r', linestyle='--', label=f'Median: {median_value}')
plt.xlabel('Meteorites number / k value')
plt.ylabel('Probability value')
plt.title('Meteorites falling on an ocean')
plt.legend()
plt.grid(True)
plt.show()
