import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad


# Define the PDF function
def pdf(y):
    return 3.2 * y * np.exp(-4 * y ** 2) + 5.9 * y * np.exp(-5 * y ** 2)


# Calculate the probability of waiting between 2 and 4 hours
prob, _ = quad(pdf, 2, 4)

# Print the probability
print(f"Probability of waiting between 2 and 4 hours: {prob:.4f}")

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
