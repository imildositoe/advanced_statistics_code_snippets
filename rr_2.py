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
plt.figure(figsize=(10, 5))
plt.scatter(x, y, color='k', label='Y Data Points')

# Here we create the design matrix using the Numpy vander function with the degree 11
# Then we scale our data to mitigate numerical instability as well as correlation
# And we perform the Ordinary Least Squares estimation and store in the beta_ols
X = np.vander(x, 11, increasing=True)
scaler = StandardScaler()
scaled_X = scaler.fit_transform(X)
beta_ols = np.linalg.inv(X.T @ X) @ X.T @ y

# Here we fit our Ordinary Least Squares model
# Then we plot our Ordinary Least Squares fit in the graph
# And we print the coefficients in the console
x_fit = np.linspace(min(x), max(x), 1000)
X_fit = np.vander(x_fit, 11, increasing=True)
y_fit_ols = X_fit @ beta_ols
plt.plot(x_fit, y_fit_ols, color='r', label='OLS Fit')
print('Coefficients OLS: \n', beta_ols)

# Before proceeding with the Ridge-regularization, we choose our lambda parameter using Cross-Validation
# For that we define the range [-5; 100] to be looped to find the best lambda value
# Then we choose 5 as the number of folds for CV, and we run the double-loop to find the optimal lambda value
# And we then print the CV lambda
lambda_values = np.logspace(-5, 5, 100)
k = 5
kf = KFold(n_splits=k)
validat_errors = []

for lambda_ in lambda_values:
    fold_errors = []
    for train_index, val_index in kf.split(scaled_X):
        X_train, X_val = scaled_X[train_index], scaled_X[val_index]
        y_train, y_val = y[train_index], y[val_index]
        model = Ridge(alpha=lambda_)
        model.fit(X_train, y_train)
        y_predicted = model.predict(X_val)
        fold_errors.append(mean_squared_error(y_val, y_predicted))
    validat_errors.append(np.mean(fold_errors))
cv_lambda = lambda_values[np.argmin(validat_errors)]
print('Cross-Validated Lambda value: ', cv_lambda)

# Then we perform the Ridge estimation using the Cross-Validated (the best lambda value)
beta_ridge = np.linalg.inv(X.T @ X + cv_lambda * np.identity(11)) @ X.T @ y

# Then we perform the Ridge fit, and we plot in our graph
# And finally we display the final graph with the Scatter, OLS fit, and Ridge fit
y_fit_ridge = X_fit @ beta_ridge
plt.plot(x_fit, y_fit_ridge, color='g', label='Ridge Fit')
print('Coefficients Ridge: \n', beta_ridge)

plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('Scatter, OLS and Ridge Regularization Fit')
plt.show()
