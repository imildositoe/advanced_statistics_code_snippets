import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
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

# Converting polynomial into linear regression using Pipeline
pipeline = Pipeline([
    ('poly', PolynomialFeatures(degree=10, include_bias=False)),
    ('linear', LinearRegression())
])
pipeline.fit(x, y)

# Applying the prediction line
x_values_plot = np.linspace(min(x) - 1, max(x) + 1, 400).reshape(-1, 1)
y_values_plot = pipeline.predict(x_values_plot)

# Visualizing the points jointly with the prediction line
plt.scatter(x, y, color='black', label='Data Points')
plt.plot(x_values_plot, y_values_plot, label='OLS Predicted Values')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
