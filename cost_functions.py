import numpy as np
import cvxpy as cp
from abc import ABC, abstractmethod

#Reference: https://github.com/staretgicclfdark/strategic_rep

class CostFunction(ABC): # why is this needed???
    @abstractmethod
    def __call__(self, z: np.array, x: np.array):
        '''

        :param z: Feature vector that player might want to have
        :param x: Feature that player now has.
        :return: the cost that player pays to become z
        '''
        pass

    def maximize_features_against_binary_model(self, x: np.array, trained_model, use_spare_cost=False):
        '''

        :param x: current vector features.
        :param trained_model: binary model that is trained and player want to get positive score on it.
        :param use_spare_cost: if we want to use some of the spare cost in order to improve player score on the trained model
        :return: vector features  that has minimum cost and get positive score on trained_model.
        '''
        pass

class MixWeightedLinearSumSquareCostFunction(CostFunction):
    """
    This class represents cost function that Consists of two parts. First part is weighted linear function
    and the second part is sum square cost function. That means calculation in the form of:
    (1-epsilon) * max {<a, x'-x>, 0} + epsilon * square(norm2(x'-x)).
    where a is the weights vector, x'-x is the change that player pays on and epsilon is the weight of the
    l2 cost function.
    """
    def __init__(self, alpha: np.array, epsilon=0.3):
        '''

        :param alpha: Weights vector. Each entry i in the vector represents the payment of moving
        one unit in the i'th feature.
        :param epsilon: The weight of the l2 cost function.
        :param cost_factor: This parameter determines the scale of the cost function. This is a const that
        multiply the cost result.
        :param spare_cost: How much palyer agree to pay more in order to be beyond the classifier bound.
        '''
        self.a = alpha
        self.epsilon = epsilon

    def __call__(self, z: np.array, x: np.array):
        cost_value = (1 - self.epsilon) * self.a @ (z - x) + self.epsilon * np.sum((z - x) ** 2)
        print("hihi")
        print((1 - self.epsilon) * self.a.T @ (z - x))
        print(self.epsilon * np.sum((z - x) ** 2))
        return max(cost_value, 0)

