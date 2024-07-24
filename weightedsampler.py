import numpy as np

class WeightedSampler:
    def __init__(self, X, sigma):
        """
        Initializes the WeightedSampler with the population X and bandwidth parameter sigma.

        Parameters:
        X (numpy array): The population from which to sample.
        sigma (float): The bandwidth parameter for the Gaussian kernel.
        """
        self.X = X
        self.sigma = sigma

    def compute_distances(self, x):
        """
        Computes the Euclidean distances between a given point x and the population X.

        Parameters:
        x (numpy array): The point from which distances are calculated.

        Returns:
        numpy array: The distances between x and each point in X.
        """
        return np.linalg.norm(self.X - x, axis=1)

    def gaussian_weights(self, distances):
        """
        Computes weights using a Gaussian kernel based on distances.

        Parameters:
        distances (numpy array): The distances between x and each point in X.

        Returns:
        numpy array: The weights based on the Gaussian kernel.
        """
        return np.exp(- (distances ** 2) / (2 * self.sigma ** 2))

    def normalize_weights(self, weights):
        """
        Normalizes the weights to sum to 1.

        Parameters:
        weights (numpy array): The weights to be normalized.

        Returns:
        numpy array: The normalized weights.
        """
        return weights / np.sum(weights)

    def sample(self, x, num_samples=1):
        """
        Samples instances from the population X with higher probability for instances near x.

        Parameters:
        x (numpy array): The point near which to sample instances.
        num_samples (int): The number of samples to draw.

        Returns:
        numpy array: The sampled instances.
        """
        distances = self.compute_distances(x)
        weights = self.gaussian_weights(distances)
        probabilities = self.normalize_weights(weights)
        indices = np.random.choice(len(self.X), size=num_samples, p=probabilities, replace=False)
        return self.X[indices]
