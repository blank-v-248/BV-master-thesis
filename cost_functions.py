import numpy as np

#Source: https://github.com/staretgicclfdark/strategic_rep


class MixWeightedLinearSumSquareCostFunction():
    """
    Cost function combining a weighted linear component and a squared L2 norm penalty.

    The function is calculated as:
        (1 - epsilon) * max{ <a, x' - x>, 0 } + epsilon * ||x' - x||²

    where:
        - a is a weight vector,
        - x' - x represents the feature modification,
        - epsilon controls the balance between linear and quadratic cost terms.

    Attributes:
        alpha (np.array): Weight vector indicating the cost per unit change for each feature.
        epsilon (float): Weight of the L2 cost function.
    """
    def __init__(self, alpha: np.array, epsilon=0.3):
        """
        Initializes the cost function with given weight parameters.

        Args:
            alpha (np.array): Vector of weights representing the cost per unit feature change.
            epsilon (float, optional): Weight of the L2 cost function. Defaults to 0.3.
        """
        self.a = alpha
        self.epsilon = epsilon

    def __call__(self, z: np.array, x: np.array):
        dim =x.shape[0]
        if dim != 1:
            cost_value = (1 - self.epsilon) * self.a.T @ (z - x) + self.epsilon * np.sum((z - x) ** 2)
        else: # if data dimension is 1, different syntax has to be used
            cost_value = (1 - self.epsilon) * self.a @ (z - x) + self.epsilon * np.sum((z - x) ** 2)
        return max(cost_value[0], 0)

