import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
import numpy as np

class ClassifierPlotter:
    def __init__(self, X, y, feature_names=["Feature 1", "Feature 2"]):
        self.X = X
        self.y = y
        self.feature_names=feature_names

    def plot_orig_dataset(self, title="Original Dataset", ):
        plt.figure(figsize=(8, 6))

        # Define colors for the two labels
        colors = ['#6FCFF5', '#EA0000']

        # Get the unique labels
        unique_labels = np.unique(self.y)

        # Map the labels to colors
        label_to_color = {unique_labels[0]: colors[0], unique_labels[1]: colors[1]}

        # Scatter plot with different colors for the two labels
        for label in unique_labels:
            plt.scatter(self.X[self.y == label, 0], self.X[self.y == label, 1],
                        label=label, color=label_to_color[label], edgecolor='k', s=100)

        # Set labels and title
        plt.xlabel(self.feature_names[0])
        plt.ylabel(self.feature_names[1])
        plt.title(title)

        # Add a legend
        plt.legend(title="Labels")

        # Show the plot
        plt.show()

    def plot_decision_surface(self, clf, title="Decision surface"):
        """
         Plots the decision surface of a fitted classifier.

         Parameters:
         clf: Fitted classifier with a predict method
         title: Title for the plot
         feature_names: Names of the features for axis labels
         """
        if not hasattr(clf, "predict"):
            raise ValueError("The classifier does not have a predict method. Please provide a fitted classifier.")

        # Create a mesh grid for the 2D feature space
        x_min, x_max = self.X[:, 0].min() - 1, self.X[:, 0].max() + 1
        y_min, y_max = self.X[:, 1].min() - 1, self.X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))

        # Predict for each point in the mesh grid
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        # Plot the decision boundary and the training points
        plt.figure(figsize=(8, 6))
        plt.contourf(xx, yy, Z, alpha=0.3)
        plt.scatter(self.X[:, 0], self.X[:, 1], c=self.y, edgecolors='k', marker='o')
        plt.title(title)
        plt.xlabel(self.feature_names[0])
        plt.ylabel(self.feature_names[1])

        plt.show()

