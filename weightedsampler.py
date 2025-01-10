import numpy as np

class WeightedSampler: #TO-DO: make sure that there is a sample from both labels!
    def __init__(self, X_train, sigma, y_train, random_seed=24):
        """
        Initializes the WeightedSampler with the population X and bandwidth parameter sigma.

        Parameters:
        X (numpy array): The population from which to sample.
        sigma (float): The bandwidth parameter for the Gaussian kernel.
        """
        self.X_train = X_train
        self.sigma = sigma
        self.y_train=y_train
        self.random_seed=random_seed

    def compute_distances(self, x):
        """
        Computes the Euclidean distances between a given point x and the population X.

        Parameters:
        x (numpy array): The point from which distances are calculated.

        Returns:
        numpy array: The distances between x and each point in X.
        """
        return np.linalg.norm(self.X_train - x, axis=1)

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
        np.random.seed(self.random_seed)
        indices = np.random.choice(len(self.X_train), size=num_samples, p=probabilities, replace=False)

        # Making sure to have all samples:
        sampled_y=self.y_train[indices]
        # Check if there's no `1` label in sampled_y
        if 1 not in sampled_y:
            # Extract all indices where y_train == 1
            positive_indices = np.where(self.y_train == 1)[0]

            # Calculate Euclidean distances to all positive samples
            distances = np.linalg.norm(self.X_train[positive_indices] - x, axis=1)

            # Find the index of the closest positive sample
            closest_positive_index = positive_indices[np.argmin(distances)]

            # Add the closest positive index to the list of indices
            indices.append(closest_positive_index)
        if 0 not in sampled_y:
            # Extract all indices where y_train == 1
            negative_indices = np.where(self.y_train == 0)[0]

            # Calculate Euclidean distances to all positive samples
            distances = np.linalg.norm(self.X_train[negative_indices] - x, axis=1)

            # Find the index of the closest positive sample
            closest_negative_index = negative_indices[np.argmin(distances)]

            # Add the closest positive index to the list of indices
            indices.append(closest_negative_index)

        return indices, self.X_train[indices]