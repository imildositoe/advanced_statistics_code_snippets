import numpy as np
import matplotlib.pyplot as plt

# Given data points
data = [
    (-3, -470285.75), (-13, -896973342427.97), (16, -5969502572609.42),
    (-5, -72645568.04), (13, -741877267973.1), (-19, -37137459427225.83),
    (12, -340017811793.51), (-17, -13102338007357.26), (-2, -9823.26),
    (6, -298898753.66), (1, 24.8), (9, -17875424305.59), (-14, -1969295907622.04),
    (-10, -69515637329.79), (11, -143863543867.51), (0, -5.99),
    (5, -45652790.26), (4, -4104126.5), (18, -20362761303926.95),
    (-7, -2063362968.96), (8, -5555367334.52), (14, -1525579228478.32)
]

# Separate data into x and y
x = np.array([point[0] for point in data])
y = np.array([point[1] for point in data])

# Scatter plot of original data
plt.scatter(x, y, color='blue', label='Data Points')

# Create the design matrix for a 10th degree polynomial
X = np.vander(x, 11, increasing=True)

# OLS estimation
a_ols = np.linalg.inv(X.T @ X) @ X.T @ y

# Ridge regularization parameter
lambda_ = 1e-5  # Adjust lambda as necessary

# Ridge estimation
a_ridge = np.linalg.inv(X.T @ X + lambda_ * np.identity(11)) @ X.T @ y

# Plot OLS fit
x_fit = np.linspace(min(x), max(x), 1000)
X_fit = np.vander(x_fit, 11, increasing=True)
y_fit_ols = X_fit @ a_ols
# plt.plot(x_fit, y_fit_ols, color='red', label='OLS Fit')

# Plot Ridge fit
y_fit_ridge = X_fit @ a_ridge
# plt.plot(x_fit, y_fit_ridge, color='green', label='Ridge Fit')

plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('Polynomial Fit with OLS and Ridge Regularization')
plt.show()
