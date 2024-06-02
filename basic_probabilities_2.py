import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad


def main():
    # We first define the probability density function pdf
    def pdf(y):
        return (10 / 99) * np.exp(-5 * y ** 2) * (32 * np.exp(y ** 2) + 59) * y

    # Then we calculate the probability of waiting between 2 and 4 hours using the definite integral function quad
    # We print the calculated probability by taking only the first item prob[0]
    # We exclude the second which is the absolute error, as quad returns an array of probability and absolute error
    print("The probability of waiting between 2 and 4 hours is: ", quad(pdf, 2, 4)[0])

    # In order to calculate our probabilities, we will first create a range of our independent values
    # We will create 500 values between 0 and 4 using linspace function following by determining the probabilities
    y_values = np.linspace(0, 4, 500)
    prob_values = pdf(y_values)

    # Now we can plot and display the pdf graph
    plt.figure(figsize=(10, 5))
    plt.plot(y_values, prob_values, label='PDF')
    plt.xlabel('Time in hours')
    plt.ylabel('Probability value')
    plt.title('Probability Density Function Graph')
    plt.legend()
    plt.grid(True)
    plt.show()

    # In order to display the histogram, we will generate 120 values between 2 and 4, which corresponds to 4h-2h=2hours
    # Then we will determine the probabilities in each probability value
    # Then divide by 60 to obtain the actual probability per given minute
    y_values_minutes = np.linspace(2, 4, 120)
    prob_values_minutes = pdf(y_values_minutes)
    prob_per_minute = prob_values_minutes/60

    # Now we can plot the histogram displaying each bar with the probability of being in that minute
    plt.figure(figsize=(10, 5))
    plt.bar(y_values_minutes, prob_per_minute, width=1/60, edgecolor='black', alpha=0.7)
    plt.xlabel('Time in hours')
    plt.ylabel('Probability value/minute')
    plt.title('Probability/minute Histogram')
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    main()
