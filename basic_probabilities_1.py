import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson

# Parameters
lambda_ = 71

# Calculate probabilities until they drop below 0.5%
k_values = np.arange(0, 140)
probabilities = poisson.pmf(k_values, lambda_)

# Determine the range of k where probabilities are above 0.5%
threshold = 0.005
valid_k = k_values[probabilities >= threshold]

print('K Values: ', k_values)
print('--------------------------------')
print('Probabilities: ', probabilities)
print('--------------------------------')
print('Threshold: ', threshold)
print('--------------------------------')
print('Valid Ks: ', valid_k)
print('--------------------------------')


# Plotting
plt.figure(figsize=(12, 6))
plt.bar(valid_k, probabilities[probabilities >= threshold], color='skyblue', alpha=0.7, label='P(X=k)')
plt.xlabel('Number of Meteorites (k)')
plt.ylabel('Probability')
plt.title('Poisson Distribution: Meteorites Falling on an Ocean')
plt.axvline(x=lambda_, color='red', linestyle='--', label=f'Expectation (E[X] = {lambda_})')

# Calculate and plot the median
cdf = poisson.cdf(k_values, lambda_)
median = np.where(cdf >= 0.5)[0][0]
plt.axvline(x=median, color='green', linestyle='--', label=f'Median (median = {median})')

# Show legend
plt.legend()
plt.show()

# Expectation
expectation = lambda_
print(f'Expectation (E[X]): {expectation}')

# Median
print(f'Median: {median}')
