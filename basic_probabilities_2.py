import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad


def solve1():
    # We first define the probability density function pdf
    def pdf(y):
        return (10/99) * np.exp(-5 * y ** 2) * (32 * np.exp(y ** 2) + 59) * y

    # Then we calculate the probability of waiting between 2 and 4 hours using the definite integral function quad
    # We print the calculated probability by taking only the first item prob[0]
    # We exclude the second which is the absolute error, as quad returns an array of probability and absolute error
    print("The probability of waiting between 2 and 4 hours is: ", quad(pdf, 2, 4)[0])

    # Define the range for y
    y = np.linspace(0, 5, 1000)

    # Calculate the PDF values
    pdf_values = pdf(y)

    # Plot the PDF
    plt.figure(figsize=(10, 5))
    plt.plot(y, pdf_values, label='PDF')
    plt.xlabel('Time (hours)')
    plt.ylabel('Probability Density')
    plt.title('Probability Density Function')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Calculate the histogram
    y_minutes = np.linspace(2, 4, (4 - 2) * 60)
    pdf_minutes = pdf(y_minutes)
    probabilities_per_minute = pdf_minutes * (1 / 60)

    # Plot the histogram
    plt.figure(figsize=(10, 5))
    plt.bar(y_minutes, probabilities_per_minute, width=1 / 60, edgecolor='black', alpha=0.7)
    plt.xlabel('Time (hours)')
    plt.ylabel('Probability per minute')
    plt.title('Histogram of Probability per Minute')
    plt.grid(True)
    plt.show()


def solve2():
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.integrate import quad

    # Define the survival function
    def S(y):
        return (40 / 99) * np.exp(-4 * y ** 2) + (59 / 99) * np.exp(-5 * y ** 2)

    # Define the PDF by taking the negative derivative of the survival function
    def f(y):
        dS_dy = np.gradient(S(y), y)
        return -dS_dy

    # Define the CDF
    def F(y):
        return 1 - S(y)

    # Calculate the probability between 2 and 4 hours
    prob, _ = quad(f, 2, 4)
    print(f"Probability of waiting between 2 and 4 hours: {prob}")

    # Plot the PDF
    y_values = np.linspace(0, 5, 500)
    pdf_values = f(y_values)

    plt.plot(y_values, pdf_values, label='PDF')
    plt.xlabel('Time (hours)')
    plt.ylabel('Density')
    plt.title('Probability Density Function')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot the histogram where each bar represents the probability of being in that minute
    minutes = np.arange(0, 301) / 60  # Convert minutes to hours
    pdf_values_at_minutes = f(minutes)
    probabilities = pdf_values_at_minutes * (1 / 60)  # Convert density to probability for each minute

    plt.bar(minutes, probabilities, width=1 / 60, edgecolor='black')
    plt.xlabel('Time (hours)')
    plt.ylabel('Probability')
    plt.title('Histogram of Probabilities per Minute')
    plt.grid(True)
    plt.show()


def main():
    solve1()
    # solve2()


if __name__ == '__main__':
    main()
