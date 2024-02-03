import matplotlib.pyplot as plt


def proportions_viz():
    # Data
    outcomes = ['For', 'Against']
    proportions = [0.71, 0.29]  # Proportions of 'For' and 'Against' outcomes

    # Create bar chart
    plt.bar(outcomes, proportions, color=['pink', 'purple'])

    # Add labels and title
    plt.xlabel('Outcome')
    plt.ylabel('Proportion')
    plt.title('Proportion of "For" and "Against" Outcomes')

    # Show plot
    plt.show()


if __name__ == '__main__':
    proportions_viz()

