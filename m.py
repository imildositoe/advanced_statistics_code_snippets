def print_hi(name):
    print(f'Hi, {name}')


if __name__ == '__main__':
    # import numpy as np
    # import matplotlib.pyplot as plt
    # from scipy.stats import linregress
    #
    # # Define the given points
    # points = np.array([
    #     (-3, -470285.75), (-13, -896973342427.97), (16, -5969502572609.42), (-5, -72645568.04),
    #     (13, -741877267973.1), (-19, -37137459427225.83), (12, -340017811793.51), (-17, -13102338007357.26),
    #     (-2, -9823.26), (6, -298898753.66), (1, 24.8), (9, -17875424305.59), (-14, -1969295907622.04),
    #     (-10, -69515637329.79), (11, -143863543867.51), (0, -5.99), (5, -45652790.26), (4, -4104126.5),
    #     (18, -20362761303926.95), (-7, -2063362968.96), (8, -5555367334.52), (14, -1525579228478.32)
    # ])
    #
    # # Extract x and y coordinates
    # x = points[:, 0]
    # y = points[:, 1]
    #
    # # Perform linear regression
    # slope, intercept, _, _, _ = linregress(x, y)
    #
    # # Generate the function
    # function = f'y = {slope:.2f}x + {intercept:.2f}'
    #
    # # Plot the points and the regression line
    # plt.scatter(x, y, label='Data Points')
    # plt.plot(x, slope * x + intercept, color='red', label='Linear Regression')
    # plt.xlabel('X')
    # plt.ylabel('Y')
    # plt.title('Linear Regression')
    # plt.legend()
    # plt.grid(True)
    # plt.show()
    #
    # print("Linear Regression Function:", function)

    # ------------------------------

    # import numpy as np
    # from sklearn.linear_model import Ridge
    # from sklearn.preprocessing import PolynomialFeatures
    # from sklearn.metrics import mean_squared_error
    #
    # # Define the data
    # x = np.array([-3, -13, 16, -5, 13, -19, 12, -17, -2, 6, 1, 9, -14, -10, 11, 0, 5, 4, 18, -7, 8, 14])
    # y = np.array([-470285.75, -896973342427.97, -5969502572609.42, -72645568.04, -741877267973.1, -37137459427225.83,
    #               -340017811793.51, -13102338007357.26, -9823.26, -298898753.66, 24.8, -17875424305.59,
    #               -1969295907622.04, -69515637329.79, -143863543867.51, -5.99, -45652790.26, -4104126.5,
    #               -20362761303926.95, -2063362968.96, -5555367334.52, -1525579228478.32])
    #
    # # Reshape x to be a 2D array
    # x = x.reshape(-1, 1)
    #
    # # Create polynomial features
    # poly = PolynomialFeatures(degree=10)
    # X_poly = poly.fit_transform(x)
    #
    # # Fit the OLS model
    # ols_model = np.linalg.lstsq(X_poly, y, rcond=None)[0]
    #
    # # Fit the Ridge model
    # ridge_model = Ridge(alpha=1.0)  # Set regularization parameter here
    # ridge_model.fit(X_poly, y)
    # ridge_coefs = ridge_model.coef_
    #
    # # Print the coefficients
    # print("OLS coefficients:", ols_model)
    # print("Ridge coefficients:", ridge_coefs)
    #
    # # Calculate the mean squared error for each model
    # ols_preds = np.dot(X_poly, ols_model)
    # ridge_preds = ridge_model.predict(X_poly)
    #
    # ols_mse = mean_squared_error(y, ols_preds)
    # ridge_mse = mean_squared_error(y, ridge_preds)
    #
    # print("OLS MSE:", ols_mse)
    # print("Ridge MSE:", ridge_mse)

    import matplotlib.pyplot as plt

    # Define the data
    x = [-3, -13, 16, -5, 13, -19, 12, -17, -2, 6, 1, 9, -14, -10, 11, 0, 5, 4, 18, -7, 8, 14]
    y = [-470285.75, -896973342427.97, -5969502572609.42, -72645568.04, -741877267973.1, -37137459427225.83,
         -340017811793.51, -13102338007357.26, -9823.26, -298898753.66, 24.8, -17875424305.59, -1969295907622.04,
         -69515637329.79, -143863543867.51, -5.99, -45652790.26, -4104126.5, -20362761303926.95, -2063362968.96,
         -5555367334.52, -1525579228478.32]

    # Create the scatter plot
    plt.scatter(x, y)
    plt.title('Scatter plot of given points')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

    # -------------------------

    import matplotlib.pyplot as plt

    # Points given in the question
    points = [
        (-3, -470285.75), (-13, -896973342427.97), (16, -5969502572609.42),
        (-5, -72645568.04), (13, -741877267973.1), (-19, -37137459427225.83),
        (12, -340017811793.51), (-17, -13102338007357.26), (-2, -9823.26),
        (6, -298898753.66), (1, 24.8), (9, -17875424305.59),
        (-14, -1969295907622.04), (-10, -69515637329.79), (11, -143863543867.51),
        (0, -5.99), (5, -45652790.26), (4, -4104126.5),
        (18, -20362761303926.95), (-7, -2063362968.96), (8, -5555367334.52),
        (14, -1525579228478.32)
    ]

    x, y = zip(*points)  # This separates the x and y coordinates into two lists

    plt.figure(figsize=(10, 6))

    # Since the y-values vary greatly, we focus on just plotting the points to visualize their distribution
    plt.scatter(x, y, color='blue')

    plt.title('Point Distribution')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True)
    plt.yscale('symlog')  # Using a symmetric log scale to handle both positive and negative values

    plt.show()




