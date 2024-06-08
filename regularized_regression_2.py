import numpy as np

# Given data points
data = [(-3, -470285.75), (-13, -896973342427.97), (16, -5969502572609.42), (-5, -72645568.04),
        (13, -741877267973.1), (-19, -37137459427225.83), (12, -340017811793.51),
        (-17, -13102338007357.26), (-2, -9823.26), (6, -298898753.66), (1, 24.8),
        (9, -17875424305.59), (-14, -1969295907622.04), (-10, -69515637329.79),
        (11, -143863543867.51), (0, -5.99), (5, -45652790.26), (4, -4104126.5),
        (18, -20362761303926.95), (-7, -2063362968.96), (8, -5555367334.52), (14, -1525579228478.32)]

# Extracting x and y values
x = np.array([point[0] for point in data])
y = np.array([point[1] for point in data])

# Constructing the design matrix X
X = np.vander(x, N=11, increasing=True)

# OLS estimate
a_ols = np.linalg.inv(X.T @ X) @ X.T @ y

# Ridge-regularized estimate
lambda_reg = 1.0  # Regularization parameter, can be adjusted
I = np.eye(11)
a_ridge = np.linalg.inv(X.T @ X + lambda_reg * I) @ X.T @ y

print("OLS coefficients:", a_ols)
print("Ridge-regularized coefficients:", a_ridge)
