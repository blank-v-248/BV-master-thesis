import numpy as np
import cvxpy as cp
from scipy.optimize import minimize
from cost_functions import MixWeightedLinearSumSquareCostFunction


class BestResponse:
    """Best response function for agents given classifier.

    Parameters
    ----------
    X: np.array
        training data matrix
    strat_features: list
        list of feature indices that can be manipulated strategically, other features remain fixed
    """

    def __init__(self, X, strat_features):
        self.X = X
        self.strat_features = strat_features

    def algorithm2(self, alpha, model, t=1, epsilon=0, mod_type="dec_f", treshold=0.5):
        ### finds optimal response for the agent given a classifier and costs
        # alpha: cost vector of changing the features
        # model: classifier that has either a .decision_function or a .predict_proba option
        # t: the amount of gaming, the higher t, the more users are willing to pay
        # epsilon: the weight of the quadratic weight in the mixed cost function
        # mod_type: what argument the model has that returns a continous
        # treshold: the threshold of the model over which it predicts +1

        n = self.X.shape[0]
        m = len(self.strat_features)
        X_strat = np.copy(self.X)
        self.costs = np.zeros(n)

        for i in range(n): #iterate over all instances
            if model.predict(X_strat[i,].reshape(1, -1)) != 1:  # people only change, if they get predicted as -1
                x0_strat = X_strat[
                    i, self.strat_features]  # these are the feautures that agent can change in the original state

                # Define the cost function by class:
                cost_func = MixWeightedLinearSumSquareCostFunction(alpha, epsilon)

                # Define the objective function and constraint: minimize costs while getting better decision
                objective = lambda x: cost_func(x, x0_strat)  # mixed cost should be minimized

                def constraint_function(x, X_strat=X_strat):# so that the prediction of the model is 1 = wortwhile
                    # f(x')=1
                    # OUTPUT NEEDS TO BE CONTINOUS!
                    X_i = np.copy(X_strat[i,])
                    np.put(X_i, self.strat_features, x)
                    if mod_type == "dec_f":
                        return model.decision_function(X_i.reshape(1, -1)) - 1
                    else:
                        return model.predict_proba(X_i.reshape(1, -1))[0, 1] - treshold

                cons_equations = [
                    {'type': 'ineq', 'fun': constraint_function},
                ]

                result = minimize(objective, x0_strat, constraints=cons_equations)
                print("--")
                # Solve the optimization problem with scipy minimize:
                # Check if the optimization was successful
                if result.success:
                    print("succes")
                    opt_strat_x = result.x.reshape(1, m)
                else:
                    print("no succes")
                    opt_strat_x = x0_strat  # Retain the original x0_strat
                print(opt_strat_x)

                # Get the cost of optimal value:
                self.costs[i] = cost_func(opt_strat_x, x0_strat)
                print("cost", self.costs[i])
                # Update X_strat based on the cost constraint: only change, if change does not cost too much
                if self.costs[i]<2*t:
                    print("smaller than", 2*t)
                    X_strat[i, self.strat_features] = opt_strat_x
                else:
                    print("not smaller than 2t")
                    self.costs[i] = 0


                self.X_shifted=np.copy(X_strat)

        ###returns:
        # X_strat: the new feautures of the agents
        return X_strat

    def get_costs(self):
        # Returns the costs of feature change
        return self.costs

    def find_differences(self):
        # Compare the elements to find differences
        differences = self.X != self.X_shifted

        # Get the indices of the differing elements
        differing_indices = np.where(differences)[0]

        return differing_indices