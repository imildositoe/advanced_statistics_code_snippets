import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.optimize import root_scalar


def main():
    # We first define the probability density function pdf
    def pdf(y):
        return (10 / 99) * np.exp(-5 * y ** 2) * (32 * np.exp(y ** 2) + 59) * y

    # Then we can define the cumulative probability function, by integrating (quad) the pfd from 0 to y
    def cdf(y):
        return quad(pdf, 0, y)[0]

    # Then we calculate the probability of waiting between 2 and 4 hours using the definite integral function quad
    # We print the calculated probability by taking only the first item prob[0]
    # We exclude the second which is the absolute error, as quad returns an array of probability and absolute error
    print("The probability of waiting between 2 and 4 hours is: ", quad(pdf, 2, 4)[0])

    # In order to calculate our probabilities, we will first create a range of our independent values
    # We will create 500 values between 0 and 4 using linspace function following by determining the probabilities
    y_values = np.linspace(0, 4, 500)
    prob_values = pdf(y_values)


    # In order to display the mean and variance, we
    # Calculate the mean (since our pdf is limited to 0;np.inf, as we are dealing with time)
    mean = quad(lambda y: y * pdf(y), 0, np.inf)[0]
    variance = (quad(lambda y: y ** 2 * pdf(y), 0, np.inf)[0]) - mean ** 2

    # To display the quartiles, we extract the root (found by approximating within the specified bracket parameter)
    # The function corresponds to cdf(y)-probability=0, where the resulting y is the quantile of the specified probability
    def get_quantile(probability):
        return root_scalar(lambda y: cdf(y) - probability, bracket=[0, 10]).root

    q1 = get_quantile(0.25)
    q2 = get_quantile(0.50)
    q3 = get_quantile(0.75)


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
    prob_per_minute = prob_values_minutes / 60

    # Now we can plot the histogram displaying each bar with the probability of being in that minute
    plt.figure(figsize=(10, 5))
    plt.bar(y_values_minutes, prob_per_minute, width=1 / 60, edgecolor='black', alpha=0.7)
    plt.xlabel('Time in hours')
    plt.ylabel('Probability value/minute')
    plt.title('Probability/minute Histogram')
    plt.grid(True)
    plt.show()


# def mean_variance_quartile():
    # Define the PDF function
    # def pdf(y):
    #     return (10 / 99) * np.exp(-5 * y ** 2) * (32 * np.exp(y ** 2) + 59) * y

    # Define the CDF function (integral of the PDF)
    # def cdf(y):
    #     return quad(pdf, 0, y)[0]

    # # Calculate the mean (since our pdf is limited to 0;np.inf, as we are dealing with time)
    # mean = quad(lambda y: y * pdf(y), 0, np.inf)[0]
    #
    # # Calculate the variance
    # mean_square = quad(lambda y: y ** 2 * pdf(y), 0, np.inf)[0]
    # variance = mean_square - mean ** 2

    # Calculate the quartiles using the CDF
    # def find_quantile(prob):
    #     result = root_scalar(lambda y: cdf(y) - prob, bracket=[0, 10])
    #     return result.root
    #
    # q1 = find_quantile(0.25)
    # median = find_quantile(0.50)
    # q3 = find_quantile(0.75)

    # Print the results
    # print(f"Mean: {mean:.4f}")
    # print(f"Variance: {variance:.4f}")
    # print(f"Q1: {q1:.4f}")
    # print(f"Median: {median:.4f}")
    # print(f"Q3: {q3:.4f}")
    #
    # # # Define the range for y
    # # y = np.linspace(0, 4, 500)
    # #
    # # # Calculate the PDF values
    # # pdf_values = pdf(y)
    #
    # # Plot the PDF with mean and quartiles
    # plt.figure(figsize=(10, 5))
    # plt.plot(y, pdf_values, label='PDF')
    # plt.axvline(mean, color='r', linestyle='--', label=f'Mean: {mean:.2f}')
    # plt.axvline(q1, color='g', linestyle='--', label=f'Q1: {q1:.2f}')
    # plt.axvline(median, color='b', linestyle='--', label=f'Median: {median:.2f}')
    # plt.axvline(q3, color='g', linestyle='--', label=f'Q3: {q3:.2f}')
    # plt.xlabel('Time (hours)')
    # plt.ylabel('Probability Density')
    # plt.title('Probability Density Function with Mean and Quartiles')
    # plt.legend()
    # plt.grid(True)
    # plt.show()


if __name__ == '__main__':
    main()