import numpy as np
from scipy.stats import poisson
import matplotlib.pyplot as plt

lambda_value = 71
threshold_value = 0.005

# Calculate probabilities until they drop below 0.5%
values_of_k = np.arange(0, 100)
probs = poisson.pmf(values_of_k, lambda_value)

# Determine the range of k where probabilities are above 0.5%
valid_k_value = values_of_k[probs >= threshold_value]

# Calculate and plot the median
cdf = poisson.cdf(values_of_k, lambda_value)
median_value = np.where(cdf >= 0.5)[0][0]
expectation_value = lambda_value

print("Expectation value: ", expectation_value)
print("Median value: ", median_value)

# Plotting
plt.figure(figsize=(10, 5))
plt.bar(valid_k_value, probs[probs >= threshold_value], color='c', alpha=0.7, label='P(X = k)')
plt.axvline(x=lambda_value, color='k', linestyle='--', label=f'Expectation: {lambda_value})')
plt.axvline(x=median_value, color='r', linestyle='--', label=f'Median: {median_value})')
plt.xlabel('Meteorites number / k value')
plt.ylabel('Probability value')
plt.title('Meteorites falling on an ocean')
plt.legend()
plt.grid(True)
plt.show()
