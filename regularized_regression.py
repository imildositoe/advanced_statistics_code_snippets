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

# --------------------------------------------------------

ridge_pipeline = Pipeline([
    ('polynomial', PolynomialFeatures(degree=11, include_bias=False)),
    ('ridge regression', Ridge(alpha=1))
])
ridge_pipeline.fit(x, y)

# Applying the prediction line using ridge
y_values_ridge = ridge_pipeline.predict(x_values_plot)

# Visualizing the points jointly with the prediction line using ridge
plt.figure(figsize=(10, 6))
plt.scatter(x, y, color='black', label='Data Points')
plt.plot(x_values_plot, y_values_ridge, label='Ridge Predicted Values')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()

# Visualizing the coefficients of the OLS and after applying ridge
print(pipeline.steps[1][1].coef_)
print('-------------------------------')
print(ridge_pipeline.steps[1][1].coef_)

print('------------------------------- Rounded--------------------------------')
a = [-2.45102755e+09, 1.54163931e+08, 1.56445499e+08, -8.25327605e+06, -2.13412020e+06, 1.22366752e+05, 4.34710995e+03, -6.37603975e+02, 5.01616160e+01, -4.96630319e+00, -1.30642987e-01]
for aa in a:
    print(np.round(aa, 2))


# b = [-2.28344677e+09, 1.50087545e+08, 1.48085753e+08, -8.13910869e+06, -2.00344222e+06, 1.21121431e+05, 3.47806954e+03, -6.32146360e+02, 5.27292402e+01, -4.97436077e+00, -1.33415648e-01]