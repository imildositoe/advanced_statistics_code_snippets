import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

# Given data in the task
y_data = [
    (-3, -470285.75), (-13, -896973342427.97), (16, -5969502572609.42),
    (-5, -72645568.04), (13, -741877267973.1), (-19, -37137459427225.83),
    (12, -340017811793.51), (-17, -13102338007357.26), (-2, -9823.26),
    (6, -298898753.66), (1, 24.8), (9, -17875424305.59), (-14, -1969295907622.04),
    (-10, -69515637329.79), (11, -143863543867.51), (0, -5.99),
    (5, -45652790.26), (4, -4104126.5), (18, -20362761303926.95),
    (-7, -2063362968.96), (8, -5555367334.52), (14, -1525579228478.32)
]

# Here we are splitting the data into x and y
x = np.array([point[0] for point in y_data])
y = np.array([point[1] for point in y_data])

# Then we display the original data points as a scatter plot
plt.scatter(x, y, color='k', label='Y Data Points')

# Here we create the design matrix using the Numpy vander function
# Then we scale our data to mitigate numerical instability as well as correlation
X = np.vander(x, 11, increasing=True)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)




# OLS estimation
a_ols = np.linalg.inv(X.T @ X) @ X.T @ y

# Plot OLS fit
x_fit = np.linspace(min(x), max(x), 1000)
X_fit = np.vander(x_fit, 11, increasing=True)
y_fit_ols = X_fit @ a_ols
print('Coefficients OLS: \n', a_ols)
plt.plot(x_fit, y_fit_ols, color='red', label='OLS Fit')



# Applying the cross-validation technique to choose the best weight value lambda_value
# Define the range of lambda values to test (avoid extremely small values)
lambdas = np.logspace(-5, 5, 100)

# Number of folds for cross-validation
k = 5
kf = KFold(n_splits=k)

# To store the average validation errors for each lambda
validation_errors = []

for lambda_ in lambdas:
    fold_errors = []

    for train_index, val_index in kf.split(X_scaled):
        X_train, X_val = X_scaled[train_index], X_scaled[val_index]
        y_train, y_val = y[train_index], y[val_index]

        # Train Ridge regression model
        model = Ridge(alpha=lambda_)
        model.fit(X_train, y_train)

        # Predict on validation set
        y_val_pred = model.predict(X_val)

        # Calculate validation error (mean squared error)
        fold_errors.append(mean_squared_error(y_val, y_val_pred))

    # Average validation error for this lambda
    validation_errors.append(np.mean(fold_errors))

# Find the optimal lambda with the lowest validation error
optimal_lambda = lambdas[np.argmin(validation_errors)]
print(f'Optimal Lambda: {optimal_lambda}')


# Ridge regularization parameter
# lambda_ = 1e-5  # Adjust lambda as necessary

# Ridge estimation
a_ridge = np.linalg.inv(X.T @ X + optimal_lambda * np.identity(11)) @ X.T @ y

# Plot Ridge fit
y_fit_ridge = X_fit @ a_ridge
print('Coefficients Ridge: \n', a_ridge)
plt.plot(x_fit, y_fit_ridge, color='green', label='Ridge Fit')

plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('Polynomial Fit with OLS and Ridge Regularization')
plt.show()
