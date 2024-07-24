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


class WeightedSampler2:
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

    def sample_all_negative(self, x_train, y_train_pred, num_samples=1):
        """
        Samples instances for all x_train instances that have a negative prediction in y_pred_shifted.

        Parameters:
        x_train (numpy array): The dataset from which to sample.
        y_pred_shifted (numpy array): The shifted predictions corresponding to x_train.
        num_samples (int): The number of samples to draw for each negative instance.

        Returns:
        list of numpy arrays: The sampled instances for each negative instance in x_train.
        """
        negative_indices = np.where(y_train_pred < 0)[0]
        all_sampled_instances = []

        for idx in negative_indices:
            x = x_train[idx]
            sampled_instances = self.sample(x, num_samples=num_samples)
            all_sampled_instances.append(sampled_instances)

        return all_sampled_instances