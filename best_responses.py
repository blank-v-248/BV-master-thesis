# TO-DO: apply strategic features everywhere!

import numpy as np
from scipy.optimize import minimize, NonlinearConstraint
from cost_functions import MixWeightedLinearSumSquareCostFunction
from weightedsampler import *
from sklearn.svm import LinearSVC
from lime.lime_tabular import LimeTabularExplainer
from plotting import *
import cvxpy as cp

def difference_finder(X0, X1):
    """
    Finds the indices of rows where any element differs between two input matrices along axis 1.
    The function compares rows of X0 and X1 element-wise.
    If any element in a row differs, the row index is returned in the output.

    Is used to find diffeences between original and shifted user dataset.

    Parameters:
        X0 (np.ndarray): First input matrix of shape (N, D).
        X1 (np.ndarray): Second input matrix of shape (N, D).

    Returns:
        np.ndarray: A 1D array containing the indices of the rows where any element differs between X0 and X1.
    """
    differences = np.any(X0 != X1, axis=1)

    # Get the indices of the differing elements
    differing_indices = np.where(differences)

    return differing_indices[0].tolist()

class Fullinformation:
    """Best response function for agents given classifier in a full information scenario.

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

    def algorithm2(self, alpha, model, t=1, epsilon=0, mod_type="dec_f", threshold=0.5, use_cp=False):
        ### finds optimal response for the agent given a classifier and costs
        # alpha: cost vector of changing the features
        # model: classifier that has either a .decision_function or a .predict_proba option
        # t: the amount of gaming, the higher t, the more users are willing to pay
        # epsilon: the weight of the quadratic term in the mixed cost function, should be between 0 and 1
        # mod_type: what argument the model has that returns a continous
        # threshold: the threshold of the model over which it predicts +1

        n = self.X.shape[0]
        m = len(self.strat_features)
        X_strat = np.copy(self.X)
        self.costs = np.zeros(n)
        index_of_failed_opt=[]
        cp_vs_scipy=[]

        for i in range(n): #iterate over all instances
            if model.predict(X_strat[i,].reshape(1, -1)) != 1:  # people only change, if they get predicted as -1
                x0_strat = X_strat[i, self.strat_features]  # these are the features that agent can change
                # Define the cost function by class:
                cost_func = MixWeightedLinearSumSquareCostFunction(alpha, epsilon)

                # Define the objective function and constraint: minimize costs while getting +1 decision
                objective = lambda x: cost_func(x, x0_strat)  # mixed cost should be minimized

                def constraint_function(x, X_strat=X_strat, X_i=None, i=i):  # Explicitly pass i
                    if X_i is None:
                        X_i = np.copy(X_strat[i,])  # i is properly captured
                    np.put(X_i, self.strat_features, x)

                    if mod_type == "dec_f":
                        return X_i.reshape(1, -1) @ model.coef_[0] + model.intercept_ - threshold #model.decision_function(X_i.reshape(1, -1)) - threshold --they are the same
                    else:
                        return model.predict_proba(X_i.reshape(1, -1))[0, 1] - threshold

                constraints = NonlinearConstraint(constraint_function, 0, np.inf)

                #result = minimize(objective, x0_strat, constraints=constraints,options={"maxiter": 100000, "barrier_tol": 1e-6, "gtol": 1e-6},method="trust-constr")

                cons_equations = [
                    {'type': 'ineq', 'fun': constraint_function},
                ]

                # Solve the optimization problem with scipy minimize, equation 4:
                result = minimize(objective, x0_strat, constraints=cons_equations,  tol= 1e-8, options={"maxiter": 1000}, method="SLSQP") #method="trust-constr", "COBLYA" SLSQP

                if use_cp:
                    x_t = cp.Variable(len(x0_strat))
                    func_to_solve = cp.Minimize(cp.maximum((1 - epsilon) * alpha.T @ (x_t - x0_strat), 0) + epsilon *cp.sum((x_t - x0_strat) ** 2))
                    constrains = [x_t @ model.coef_[0] >= -model.intercept_ + threshold]

                    prob = cp.Problem(func_to_solve, constrains)
                    prob.solve()
                    opt_strat_x_cp=x_t.value

                # Check if the optimization was successful
                if result.success:
                    opt_strat_x = result.x #.reshape(1, m)
                    assert model.predict(opt_strat_x.reshape(1, -1)) == 1, f"Change happening, however prediction not worthwile! Problematic index: {i}"
                else:
                    opt_strat_x = x0_strat  # Retain the original x0_strat
                    index_of_failed_opt.append(i)

                if use_cp:
                    cp_vs_scipy.append(np.allclose(opt_strat_x, opt_strat_x_cp, atol=1e-2))
                    opt_strat_x = np.copy(opt_strat_x_cp)

                # Get the cost of optimal value:
                self.costs[i] = cost_func(opt_strat_x, x0_strat)

                # Update X_strat based on the cost constraint: only change, if change does not too costly = feasible
                if self.costs[i]<=2*t:
                    X_strat[i, self.strat_features] = opt_strat_x
                else:
                    self.costs[i] = 0

                self.X_shifted=np.copy(X_strat)

        #save failed indices:
        self.index_of_failed_opt=index_of_failed_opt

        if use_cp:
            # print cases CP vs Scipy:
            num_true = sum(cp_vs_scipy)  # True counts as 1, False as 0 in Python

            print(f"Number where CP and scipy give same result: {num_true}")
            #print(f"Percentage of them compared to all changes made: {percent_true:.2f}%")

        return X_strat

    def get_costs(self):
        # Returns the costs of feature change
        return self.costs

    def find_differences(self):
        # Get the indices of the differing elements
        differing_indices = difference_finder(self.X, self.X_shifted)

        return differing_indices
    def get_failed_ind(self):
        # Returns the indices where optimizer failed
        return self.index_of_failed_opt


class NoInformation:
    def __init__(self, X_test, strat_features, alpha, eps, y_test=None, plotting_ind=None):
        self.X_test = X_test
        self.strat_features = strat_features
        self.alpha=alpha
        self.eps=eps
        self.y_test=y_test
        self.plotting_ind=plotting_ind

    def algorithm4_utility(self, X_train, y_train_pred, sigma,m, t, threshold):
        ws = WeightedSampler(X_train, sigma, y_train_pred) # initialize weighted sampler
        n = (self.X_test.shape[0])
        dim = (self.X_test.shape[1])

        # create arrays to store results
        y_pred_est = np.empty(n)
        y_pred_est_after = np.empty(n)
        x_shifted = np.empty([n,dim])
        costs=np.empty(n)

        for ind, x in enumerate(self.X_test): # go over every user in the test set
            ind_c, T_c = ws.sample(x, m) # every user samples m other users from X_train
            y_c = y_train_pred[ind_c].reshape(m, 1) #users learn about the predictions in the sample

            f_est = LinearSVC(dual=False) # they estimate a basic linear model on the sample
            f_est.fit(T_c, y_c)

            y_pred_est[ind] = f_est.predict(x.reshape(1, -1)) #they estimate their prediction with the estimated model

            # They best respond as if the estimated model would be the full information model:
            bestresponse_sample=Fullinformation(x.reshape(1, -1), self.strat_features)
            x_shift = bestresponse_sample.algorithm2(self.alpha, f_est, t, self.eps, mod_type="dec_f", threshold=1e-06) # as it is always a linear model threshhold it set to this constant

            # Save the results:
            x_shifted[ind,]=x_shift[0]
            y_pred_est_after[ind]=f_est.predict(x_shifted[ind].reshape(1, -1))
            costs[ind]=bestresponse_sample.get_costs()

            #Save results for plotting:
            if ind==self.plotting_ind:
                self.x_plotting = np.vstack((x, T_c))
                self.y_plotting = np.vstack((y_pred_est[ind], y_c))
                self.y_plotting_true = self.y_test[np.hstack((ind, ind_c))]
                self.model_plotting = f_est
                self.x_shifted_plotting=np.vstack((x_shift[0], T_c))

        self.y_pred_est=np.copy(y_pred_est)
        self.y_pred_est_after = np.copy(y_pred_est_after)
        self.costs=np.copy(costs)
        self.X_shifted=np.copy(x_shifted)

        return self.X_shifted

    def compute_beta(self, x, x_p, t, i=None):
        """
        Computes the value of beta such that the cost function equals 2t, while ensuring beta is in the range [0, 1].
        """
        # Compute A and B
        A = self.eps * np.sum(( x-x_p) ** 2)
        B = (1 - self.eps) * self.alpha.T @ (x-x_p)

        # Compute the discriminant
        discriminant = B ** 2 + 4 * A * t

        # Ensure the discriminant is non-negative
        assert discriminant >= 0, f"Discriminant is negative, no real solutions for beta. Problematic index: {i}" # D has to be non-negative

        # Calculate the two possible beta values
        sqrt_discriminant = np.sqrt(discriminant)

        beta_1 = (-B - sqrt_discriminant) / (2 * A) + 1
        beta_2 = (-B + sqrt_discriminant) / (2 * A) + 1 # the larger beta should always be larger than 1 (beta=1 means no change, should be feasible)
        assert beta_2 >= 1, "Beta=1 not feasible" # if beta_2 < 1 not feasible solution
        # Pick the beta that is within [0, 1]
        if beta_1 <= 0: # if total change possible, take beta=0 (x shifts to x_p)
            return 0
        elif beta_1 > 0: # otherwise take biggest change possible
            return min(beta_1, 1)

    def algorithm4_imitation(self, X_train, y_train_pred, sigma,m, t):
        ws = WeightedSampler(X_train, sigma, y_train_pred)
        cost_func = MixWeightedLinearSumSquareCostFunction(self.alpha, self.eps)
        n = (self.X_test.shape[0])

        x_shifted = np.empty((self.X_test.shape))
        costs=[]
        for ind, x in enumerate(self.X_test):
            ind_c, T_c = ws.sample(x, m) #users take a sample from X_train
            y_c = y_train_pred[ind_c].reshape(m, 1) #users get the models decision on that sample

            # finding +1 labeled instances in the sample:
            ind_plus = np.where(y_c.flatten() == 1)[0]
            T_c_plus = T_c[ind_plus]

            x_p = T_c_plus.mean(axis=0) # mean of users with +1 decision

            beta_star=self.compute_beta(x, x_p, t, i=ind)
            x_shifted[ind, ]=beta_star*x+(1-beta_star)*x_p #shift into users direction

            costs.append(cost_func(x_shifted[ind], x))

            #Save results for plotting:
            if ind==self.plotting_ind:
                self.x_plotting = np.vstack((x, T_c))
                self.y_plotting = y_train_pred[np.hstack((ind, ind_c))]
                self.x_shifted_plotting=np.vstack((x_shifted[0], T_c))

        self.costs=np.copy(costs)
        self.X_shifted=np.copy(x_shifted)

        return self.X_shifted

    def get_costs(self):
        # Returns the costs of feature change
        return self.costs

    def find_differences(self):

        # Get the indices of the differing elements
        differing_indices =  difference_finder(self.X_test, self.X_shifted)

        return differing_indices

    def est_pred_before(self):
        # Returns the estimated model outcome of the users before the shift
        return self.y_pred_est

    def est_pred_after(self):
        # Returns the estimated model outcome of the users after the shift
        return self.y_pred_est_after

    def plot_sample(self, estimated=True, model_true=None):
        if self.plotting_ind==None:
            raise ValueError("Plotting index is none. If plotting is requested, choose an index of the test set.")
        if estimated:
            plotter=ClassifierPlotter(self.x_plotting,self.y_plotting.ravel())
            plotter.plot_decision_surface(self.model_plotting, title=f"Plot of decision surface for sample of instance {self.plotting_ind}.\nLabels: predicted labels by f, model: estimated model of user", X_shifted=self.x_shifted_plotting, highlighted_ind_circle=0)
        else:
            plotter = ClassifierPlotter(self.x_plotting, self.y_plotting_true)
            plotter.plot_decision_surface(model_true, title=f"Plot of decision surface for sample of instance {self.plotting_ind}.\nLabels: true labels, model: true f model", X_shifted=self.x_shifted_plotting,
                                          highlighted_ind_circle=0)

class PartialInformation:
    def __init__(self, x_train, x_test, strat_features):
        self.X_train = x_train
        self.X_test=x_test
        self.strat_features = strat_features

        # create arrays to store results
        self.y_pred_est = []
        self.y_pred_est_after = []
        self.costs=[]

    def parse_lime_output(self, exp_list, margin=0.001):
        """
        Parses the LIME explanation output into a structured format with feature name, lower bound, upper bound, and weight.

        Args:
            exp_list (list of tuples): A list where each element is a tuple containing a feature condition (string)
                                           and its associated weight (numeric).
            margin (float, optional): A small buffer added/subtracted to/from the feature bounds to account for
                                          precision. Default is 0.001.

        Returns:
            list: A list of parsed feature conditions in the format [feature_name, lower_bound, upper_bound, weight].
        """
        parsed_output = []

        for feature_condition, weight in exp_list: #Loops through each feature condition and weight from the input LIME explanations.
            # The feature condition string is split into its parts (e.g., 'age <= 30' becomes ['age', '<=', '30']):
            feature_parts = feature_condition.split(' ')

            lower_bound = None
            upper_bound = None

            #Based on the condition, it is determined whether it's a lower bound, upper bound, or a range of values.
            if len(feature_parts) == 3:
                feature_name = feature_parts[0]
                if '<' in feature_condition:
                    if '<=' == feature_parts[1]:
                        upper_bound = float(feature_parts[2]) + margin #It adjusts the bounds by adding/subtracting the margin for more precise limits.
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

            # The parsed data is then stored in a list with the format:
            parsed_output.append([feature_name, lower_bound, upper_bound, weight])

        return parsed_output

    def algorithm3(self, f, threshold, alpha, epsilon, budget=2, mod_type="dec_f"):
        explainer = LimeTabularExplainer( # A lime explainer is trained on X_train
            training_data=self.X_train,
            feature_names=list(range(self.X_train.shape[1])),
            class_names=['class_0', 'class_1'],
            mode='classification'
        )
        cost_func=MixWeightedLinearSumSquareCostFunction(alpha, epsilon)

        def predict_proba_linSVC(X): # This is only needed for Linear SVCs, as it does not have a predict proba function:
            decision = f.decision_function(X)
            expit = lambda x: 1 / (1 + np.exp(-x))
            proba = np.apply_along_axis(expit, 0, decision)
            return np.column_stack([1 - proba, proba]) if proba.ndim == 1 else proba

        delta_L=np.copy(self.X_test) # the shifted X is created as a copy of the original
        for i in range(self.X_test.shape[0]): # loop over every instance
            if mod_type=="dec_f":
                exp = explainer.explain_instance( # create explanation for that instance
                    data_row=self.X_test[i],
                    predict_fn=predict_proba_linSVC
                )
            else:
                exp = explainer.explain_instance( # create explanation for that instance
                    data_row=self.X_test[i],
                    predict_fn=f.predict_proba
                )
            exp_values = exp.as_list() #export explanation values
            loc_prob =exp.local_pred # export probability of prediction of LIME

            self.y_pred_est.append(0 if loc_prob < threshold else 1)

            if loc_prob<threshold: # if the probability of being a +1 is smaller than a threshold, then shift:
                exp_vector = self.parse_lime_output(exp_values) # parse the LIME values
                exp_filt = [ # filter only values that decline the probability of being a +1
                    vector for vector in exp_vector
                    if np.isin(float(vector[0]), self.strat_features) and vector[3] < 0
                ]
                exp_sorted = sorted(exp_filt, key=lambda x: x[3]) #sort so that biggest influence is first

                if len(exp_sorted) == 0: # if there are no such features, user does not change
                    continue

                # initialize total costs and new probability as old probability
                cost_cum = 0
                new_exp_loc_prob = loc_prob

                for j in range(len(exp_sorted)): #go over every feature that could improve classifiction to a +1
                    # Split LIME output:
                    feat =int(exp_sorted[j][0])
                    Z_low = exp_sorted[j][1]
                    Z_up = exp_sorted[j][2]

                    # Check costs for increasing or decreasing feature
                    if Z_low is None:
                        cost_low = np.inf
                    else:
                        cost_low = cost_func(np.atleast_1d(self.X_test[i, feat]), np.atleast_1d(Z_low))

                    if Z_up is None:
                        cost_up = np.inf
                    else:
                        cost_up = cost_func(np.atleast_1d(self.X_test[i, feat]), np.atleast_1d(Z_up))

                    if min(cost_low, cost_up) < budget - cost_cum: # if at least one is feasible, make a change
                        # choose the cheaper change
                        if cost_low<cost_up:
                            delta_L[i, feat] = Z_low
                            cost_cum += cost_low
                        else:
                            delta_L[i, feat] = Z_up
                            cost_cum += cost_up
                        # Check if the prediction is already positive:
                        if mod_type=="dec_f":
                            new_exp_loc_prob = explainer.explain_instance(data_row=delta_L[i,], predict_fn=predict_proba_linSVC).local_pred
                        else:
                            new_exp_loc_prob = explainer.explain_instance(data_row=delta_L[i,],
                                                                          predict_fn=f.predict_proba).local_pred
                        if new_exp_loc_prob >= threshold: # if it is positive, the user does not change anymore features
                            continue
                self.costs.append(cost_cum)
                self.y_pred_est_after.append(0 if new_exp_loc_prob < threshold else 1)
            else: # no change happening
                self.costs.append(0)
                self.y_pred_est_after.append(0 if loc_prob < threshold else 1)
        self.X_shifted=np.copy(delta_L)
        return delta_L

    def get_costs(self):
        # Returns the costs of feature change
        return self.costs

    def find_differences(self):
        # Returns the indices of instances who changed their fetaures

        # Get the indices of the differing elements
        differing_indices = difference_finder(self.X_test, self.X_shifted)

        return differing_indices

    def est_pred_before(self):
        # Returns the estimated model outcome of the users before the shift
        return self.y_pred_est

    def est_pred_after(self):
        # Returns the estimated model outcome of the users after the shift
        return self.y_pred_est_after






