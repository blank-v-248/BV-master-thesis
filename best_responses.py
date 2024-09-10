import numpy as np
import cvxpy as cp
from scipy.optimize import minimize
from cost_functions import MixWeightedLinearSumSquareCostFunction
from weightedsampler import *
from sklearn.svm import LinearSVC
from lime.lime_tabular import LimeTabularExplainer

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
                        return model.decision_function(X_i.reshape(1, -1))- treshold
                    else:
                        return model.predict_proba(X_i.reshape(1, -1))[0, 1] - treshold

                cons_equations = [
                    {'type': 'ineq', 'fun': constraint_function},
                ]

                result = minimize(objective, x0_strat, constraints=cons_equations)

                # Solve the optimization problem with scipy minimize:
                # Check if the optimization was successful
                if result.success:
                    opt_strat_x = result.x #.reshape(1, m)
                else:
                    opt_strat_x = x0_strat  # Retain the original x0_strat


                # Get the cost of optimal value:
                self.costs[i] = cost_func(opt_strat_x, x0_strat)
                # Update X_strat based on the cost constraint: only change, if change does not cost too much
                if self.costs[i]<2*t:
                    X_strat[i, self.strat_features] = opt_strat_x
                else:
                    self.costs[i] = 0


                self.X_shifted=np.copy(X_strat)

        ###returns:
        # X_strat: the new feautures of the agents
        return X_strat

    def get_costs(self):
        # Returns the costs of feature change
        return self.costs

    def find_differences(self):

        # Get the indices of the differing elements
        differing_indices = difference_finder(self.X, self.X_shifted)

        return differing_indices

    def algorithm4(self, X_train, y_train, sigma, m):
        ws = WeightedSampler(X_train, sigma)
        n=(self.X.shape[0])
        y_pred_est=np.empty(n)
        for ind, x in enumerate(self.X):
            ind_c, T_c =ws.sample(x, m)
            y_c=y_train[ind_c].reshape(m,1)

            f_est=LinearSVC(dual=False)
            f_est.fit(T_c, y_c)

            y_pred_est[ind]=f_est.predict(x.reshape(1, -1))

        return(y_pred_est)

class algorithm4:
    def __init__(self, X_test, strat_features):
        self.X = X_test
        self.strat_features = strat_features

    def sample_predict_shift_utility(self, X_train, y_train, sigma,m, alpha, t, eps, treshold):
        ws = WeightedSampler(X_train, sigma)
        n = (self.X.shape[0])
        dim = (self.X.shape[1])
        y_pred_est = np.empty(n)
        y_pred_est_after = np.empty(n)
        x_shifted = np.empty([n,dim])
        costs=np.empty(n)
        for ind, x in enumerate(self.X):
            ind_c, T_c = ws.sample(x, m)
            y_c = y_train[ind_c].reshape(m, 1)

            f_est = LinearSVC(dual=False)
            f_est.fit(T_c, y_c)

            y_pred_est[ind] = f_est.predict(x.reshape(1, -1))

            bestresponse_sample=BestResponse(x.reshape(1, -1), self.strat_features)

            x_try = bestresponse_sample.algorithm2(alpha, f_est, t, eps, mod_type="dec_f", treshold=treshold)

            x_shifted[ind,]=x_try[0]

            y_pred_est_after[ind]=f_est.predict(x_shifted[ind].reshape(1, -1))

            costs[ind]=bestresponse_sample.get_costs()

        self.y_pred_est=np.copy(y_pred_est)
        self.y_pred_est_after = np.copy(y_pred_est_after)
        self.costs=np.copy(costs)
        self.X_shifted=np.copy(x_shifted)

        return self.X_shifted

    def sample_predict_shift_imitation(self, X_train, y_train, sigma,m, beta, alpha, epsilon):
        ws = WeightedSampler(X_train, sigma)
        cost_func = MixWeightedLinearSumSquareCostFunction(alpha, epsilon)
        n = (self.X.shape[0])

        x_shifted = np.empty((self.X.shape))
        costs=np.empty(n)
        for ind, x in enumerate(self.X):
            ind_c, T_c = ws.sample(x, m)
            y_c = y_train[ind_c].reshape(m, 1)

            # finding +1 labeled samples:
            ind_plus = np.where(y_c.flatten() == 1)[0]
            T_c_plus = T_c[ind_plus]

            x_p = T_c_plus.mean(axis=0)
            x_shifted[ind, ]=beta*x+(1-beta)*x_p

            costs[ind]=cost_func(x_shifted[ind], x)

        self.costs=np.copy(costs)
        self.X_shifted=np.copy(x_shifted)

        return self.X_shifted

    def get_costs(self):
        # Returns the costs of feature change
        return self.costs

    def find_differences(self):
        # Returns the indices of instances who changed their fetaures

        # Compare the elements to find differences
        differences = self.X != self.X_shifted

        # Get the indices of the differing elements
        differing_indices = np.where(differences)[0]

        return differing_indices

    def est_pred_before(self):
        # Returns the estimated model outcome of the users before the shift
        return self.y_pred_est

    def est_pred_after(self):
        # Returns the estimated model outcome of the users after the shift
        return self.y_pred_est_after

class LimeFeatureChanger:
    def __init__(self, x_train, x_test, strat_features):
        self.X_train = x_train
        self.X_test=x_test
        self.strat_features = strat_features

    def parse_lime_output(self, exp_list, margin=0.001):
        parsed_output = []

        for feature_condition, weight in exp_list:
            # Split the condition into parts
            feature_parts = feature_condition.split(' ')

            lower_bound = None
            upper_bound = None

            if len(feature_parts) == 3:
                feature_name = feature_parts[0]
                if '<' in feature_condition:
                    if '<=' == feature_parts[1]:
                        upper_bound = float(feature_parts[2]) + margin
                    else:
                        upper_bound = float(feature_parts[2])
                else:
                    if '=>' == feature_parts[1]:
                        lower_bound = float(feature_parts[2]) - margin
                    else:
                        lower_bound = float(feature_parts[2])
            else:
                feature_name = feature_parts[2]
                if '<=' == feature_parts[1]:
                    lower_bound = float(feature_parts[0]) - margin
                else:
                    lower_bound = float(feature_parts[0])

                if '<=' == feature_parts[1]:
                    upper_bound = float(feature_parts[4]) + margin
                else:
                    upper_bound = float(feature_parts[4])

            # Create the new tuple format
            parsed_output.append([feature_name, lower_bound, upper_bound, weight])

        return parsed_output

    def explainer(self, f, threshold, alpha, epsilon, budget=10):
        explainer = LimeTabularExplainer(
            training_data=self.X_train,
            feature_names=list(range(self.X_train.shape[1])),
            class_names=['class_0', 'class_1'],
            mode='classification'
        )
        cost_func=MixWeightedLinearSumSquareCostFunction(alpha, epsilon)

        def predict_proba_linSVC(X):
            decision = f.decision_function(X)
            expit = lambda x: 1 / (1 + np.exp(-x))
            proba = np.apply_along_axis(expit, 0, decision)
            return np.column_stack([1 - proba, proba]) if proba.ndim == 1 else proba

        delta_L=np.copy(self.X_test)
        for i in range(self.X_test.shape[0]):
            exp = explainer.explain_instance(
                data_row=self.X_test[i],
                predict_fn=predict_proba_linSVC
            )
            exp_values = exp.as_list()
            loc_prob =exp.local_pred

            if loc_prob<threshold:
                exp_vector = self.parse_lime_output(exp_values)
                exp_filt = [
                    vector for vector in exp_vector
                    if np.isin(float(vector[0]), self.strat_features) and vector[3] < 0
                ]
                exp_sorted = sorted(exp_filt, key=lambda x: x[3])

                if len(exp_sorted) == 0:
                    continue
                cost_cum = 0
                for j in range(len(exp_sorted)):
                    feat =int(exp_sorted[j][0])
                    Z_low = exp_sorted[j][1]
                    Z_up = exp_sorted[j][2]
                    cost_low = cost_func(np.atleast_1d(self.X_test[i, feat]), np.atleast_1d(Z_low))
                    cost_up = cost_func(np.atleast_1d(self.X_test[i, feat]), np.atleast_1d(Z_up))
                    if min(cost_low, cost_up) < budget - cost_cum:
                        if cost_low<cost_up:
                            delta_L[i, feat] = Z_low
                            cost_cum += cost_low
                        else:
                            delta_L[i, feat] = Z_up
                            cost_cum += cost_up
                        new_exp_loc_prob = explainer.explain_instance(data_row=delta_L[i, feat], predict_fn=predict_proba_linSVC).loc_pred
                    if new_exp_loc_prob >= threshold:
                        continue
        return delta_L


    def alg1(self, alpha, eps, limeexpl, x):
        delta_l=np.copy(self.X_test)



        return None






