import matplotlib.pyplot as plt

# Assuming X is a 2D array or DataFrame of shape (n_samples, 2)
# and y_train is a 1D array of labels

def plot_orig_dataset(X, y, title="Original Dataset", feature_names=["Feature 1", "Feature 2"]):
    plt.figure(figsize=(8, 6))
    print("this is a new function")

    # Define colors for the two labels
    colors = ['#6FCFF5', '#EA0000']

    # Get the unique labels
    unique_labels = [0, 1]

    # Map the labels to colors
    label_to_color = {unique_labels[0]: colors[0], unique_labels[1]: colors[1]}

    # Scatter plot with different colors for the two labels
    for label in unique_labels:
        plt.scatter(X[y == label, 0], X[y == label, 1],
                    label=label, color=label_to_color[label], edgecolor='k', s=100)

    # Set labels and title
    plt.xlabel(feature_names[0])
    plt.ylabel(feature_names[1])
    plt.title(title)

    # Add a legend
    plt.legend(title="Labels")

    # Show the plot
    plt.show()
