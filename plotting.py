import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
import numpy as np

def difference_finder(X0, X1):
    """
    Finds the indices of rows where any element differs between two input matrices along axis 1.
    The function compares rows of X0 and X1 element-wise.
    If any element in a row differs, the row index is returned in the output.

    Parameters:
        X0 (np.ndarray): First input matrix of shape (N, D).
        X1 (np.ndarray): Second input matrix of shape (N, D).

    Returns:
        np.ndarray: A 1D array containing the indices of the rows where any element differs between X0 and X1.
    """
    differences = np.any(X0 != X1, axis=1)

    # Get the indices of the differing elements
    differing_indices = np.where(differences)

    return differing_indices

class ClassifierPlotter:
    def __init__(self, X, y, feature_names=["Feature 1", "Feature 2"]):
        self.X = X
        self.y = y
        self.feature_names=feature_names
        # Get the unique labels
        self.unique_labels = np.unique(self.y)
        # Define colors for the two labels
        colors = ['#6FCFF5', '#EA0000']


        # Map the labels to colors
        self.label_to_color = {self.unique_labels[0]: colors[0], self.unique_labels[1]: colors[1]}

    def plot_orig_dataset(self, title="Original Dataset", ):
        plt.figure(figsize=(8, 6))



        # Scatter plot with different colors for the two labels
        for label in self.unique_labels:
            plt.scatter(self.X[self.y == label, 0], self.X[self.y == label, 1],
                        label=label, color=self.label_to_color[label], edgecolor='k', s=100)

        # Set labels and title
        plt.xlabel(self.feature_names[0])
        plt.ylabel(self.feature_names[1])
        plt.title(title)

        # Add a legend
        plt.legend(title="Labels")

        # Show the plot
        plt.show()
        plt.close()

    def plot_decision_surface(self, clf, title="Decision surface", X_shifted=None, ax=None):
        """
        Plots the decision surface of a fitted classifier.

        Parameters:
        clf: Fitted classifier with a predict method
        title: Title for the plot
        X_shifted: Optional shifted data to plot arrows for differences
        ax: Matplotlib axis to plot on. If None, creates a new figure.
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

        # Use the provided axis or create a new one
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))

        # Plot the decision boundary and the training points
        ax.contourf(xx, yy, Z, alpha=0.3, colors=['#6FCFF5', '#EA0000', '#EA0000', '#EA0000'])
        for label in self.unique_labels:
            ax.scatter(self.X[self.y == label, 0], self.X[self.y == label, 1],
                       label=label, color=self.label_to_color[label], edgecolor='k', s=100)
        ax.set_title(title)
        ax.set_xlabel(self.feature_names[0])
        ax.set_ylabel(self.feature_names[1])
        ax.legend(title="Labels")

        # If X_shifted is provided, check for differences and plot arrows:
        if X_shifted is not None:
            differing_indices = difference_finder(self.X, X_shifted)
            X_diff = self.X[differing_indices]
            X_shifted_diff = X_shifted[differing_indices]
            y_labels = self.y[differing_indices]

            for i, (x1, x2) in enumerate(zip(X_diff, X_shifted_diff)):
                ax.arrow(x1[0], x1[1], x2[0] - x1[0], x2[1] - x1[1], color='orange',
                         head_width=0.05, head_length=0.1, length_includes_head=True)
                ax.scatter(x2[0], x2[1], color=self.label_to_color[y_labels[i]], edgecolor='orange', s=100)

        if ax is None:
            plt.show()
            plt.close()
