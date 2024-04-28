import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV
from sklearn.pipeline import Pipeline

# Data given in the task
data = [
    (-3, -470285.75), (-13, -896973342427.97), (16, -5969502572609.42),
    (-5, -72645568.04), (13, -741877267973.1), (-19, -37137459427225.83),
    (12, -340017811793.51), (-17, -13102338007357.26), (-2, -9823.26),
    (6, -298898753.66), (1, 24.8), (9, -17875424305.59),
    (-14, -1969295907622.04), (-10, -69515637329.79), (11, -143863543867.51),
    (0, -5.99), (5, -45652790.26), (4, -4104126.5),
    (18, -20362761303926.95), (-7, -2063362968.96), (8, -5555367334.52),
    (14, -1525579228478.32)
]

# Retrieving the given data separated into x and y arrays
x = np.array([point[0] for point in data]).reshape(-1, 1)
y = np.array([point[1] for point in data])

pipeline = Pipeline([
    ('poly', PolynomialFeatures(degree=11, include_bias=False)),
    ('linear', LinearRegression())
])
pipeline.fit(x, y)

# Applying the prediction line
x_values_plot = np.linspace(min(x) - 1, max(x) + 1, 400).reshape(-1, 1)
y_values_plot = pipeline.predict(x_values_plot)

# Visualizing the points jointly with the prediction line
plt.figure(figsize=(10, 6))
plt.scatter(x, y, color='black', label='Data Points')
plt.plot(x_values_plot, y_values_plot, label='OLS Predicted Values')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()

# Visualizing the coefficients of the OLS
print(pipeline.steps[1][1].coef_)

# --------------------------------------------------------

pipeline_ridge = Pipeline([
    ('polynomial', PolynomialFeatures(degree=11, include_bias=False)),
    ('ridge regression', Ridge(alpha=1))
])
pipeline_ridge.fit(x, y)

# Applying the prediction line using ridge
y_values_ridge = pipeline_ridge.predict(x_values_plot)

# Visualizing the points jointly with the prediction line using ridge
plt.figure(figsize=(10, 6))
plt.scatter(x, y, color='black', label='Data Points')
plt.plot(x_values_plot, y_values_ridge, label='Ridge Predicted Values')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()

# Visualizing the coefficients of ridge
print(pipeline_ridge.steps[1][1].coef_)


def ridge_fit(train, predictors, target, alpha):
    X = train[predictors].copy()
    y = train[[target]].copy()

    x_mean = X.mean()
    x_std = X.std()

    X = (X - x_mean) / x_std
    X["intercept"] = 1
    X = X[["intercept"] + predictors]

    penalty = alpha * np.identity(X.shape[1])
    penalty[0][0] = 0

    B = np.linalg.inv(X.T @ X + penalty) @ X.T @ y
    # B.index = ["intercept", "athletes", "events"]
    return B, x_mean, x_std

B, x_mean, x_std = ridge_fit(train, predictors, target, alpha)


def ridge_predict(test, predictors, x_mean, x_std, B):
    test_X = test[predictors]
    test_X = (test_X - x_mean) / x_std
    test_X["intercept"] = 1
    test_X = test_X[["intercept"] + predictors]

    predictions = test_X @ B
    return predictions

predictions = ridge_predict(test, predictors, x_mean, x_std, B)


ridge = Ridge(alpha=alpha)
ridge.fit(X[predictors], y)

# sklearn_predictions = ridge.predict(test_X[predictors])
# predictions - sklearn_predictions

from sklearn.metrics import mean_absolute_error

errors = []
alphas = [10 ** i for i in range(-2, 4)]

for alpha in alphas:
    B, x_mean, x_std = ridge_fit(train, predictors, target, alpha)
    predictions = ridge_predict(test, predictors, x_mean, x_std, B)

    errors.append(mean_absolute_error(test[target], predictions))

print(errors)



# To remove
print('-------Rounded---------')
for aa in [-2.45102755e+09, 1.54163931e+08, 1.56445499e+08, -8.25327605e+06, -2.13412020e+06, 1.22366752e+05, 4.34710995e+03, -6.37603975e+02, 5.01616160e+01, -4.96630319e+00, -1.30642987e-01]:
    print(np.round(aa, 2))
print('........')
for aa in [-2.28344677e+09, 1.50087545e+08, 1.48085753e+08, -8.13910869e+06, -2.00344222e+06, 1.21121431e+05, 3.47806954e+03, -6.32146360e+02, 5.27292402e+01, -4.97436077e+00, -1.33415648e-01]:
    print(np.round(aa, 2))
print('-------Rounded---------')
