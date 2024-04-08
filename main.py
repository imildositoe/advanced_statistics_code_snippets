def print_hi(name):
    print(f'Hi, {name}')


if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import linregress

    # Define the given points
    points = np.array([
        (-3, -470285.75), (-13, -896973342427.97), (16, -5969502572609.42), (-5, -72645568.04),
        (13, -741877267973.1), (-19, -37137459427225.83), (12, -340017811793.51), (-17, -13102338007357.26),
        (-2, -9823.26), (6, -298898753.66), (1, 24.8), (9, -17875424305.59), (-14, -1969295907622.04),
        (-10, -69515637329.79), (11, -143863543867.51), (0, -5.99), (5, -45652790.26), (4, -4104126.5),
        (18, -20362761303926.95), (-7, -2063362968.96), (8, -5555367334.52), (14, -1525579228478.32)
    ])

    # Extract x and y coordinates
    x = points[:, 0]
    y = points[:, 1]

    # Perform linear regression
    slope, intercept, _, _, _ = linregress(x, y)

    # Generate the function
    function = f'y = {slope:.2f}x + {intercept:.2f}'

    # Plot the points and the regression line
    plt.scatter(x, y, label='Data Points')
    plt.plot(x, slope * x + intercept, color='red', label='Linear Regression')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Linear Regression')
    plt.legend()
    plt.grid(True)
    plt.show()

    print("Linear Regression Function:", function)
