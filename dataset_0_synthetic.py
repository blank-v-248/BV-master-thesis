# NO IMPROVEMENT IS ALLOWED YET/NO GROUND TRUTH

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import argparse
import pandas as pd

from best_responses import *
from plotting import *
from dataset_load import *

from weightedsampler import *
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier


import warnings
from sklearn.exceptions import DataConversionWarning
import copy

# Suppress the specific warning
warnings.filterwarnings(action='ignore', category=DataConversionWarning)

class main_class:
    def __init__(self, dimension_number: int, n: int, t: float, eps : float) -> None:
        self.dimension_number = dimension_number
        self.plotname = "moons_lin.pdf"
        self.tablename = "moons_lin.csv"
        self.strat_features = np.array([0, 1])
        self.alpha = np.array([1, 1]).reshape(2, 1)
        self.t = t
        self.eps = eps
        self.feature_names = ["Feature1", "Feature2"]
        self.n=n
        self.threshold=0.5+1e-03

    def info_comparison(self, moons: bool, linear: bool):
        if moons:
            x_train, x_test, y_train, y_test = synth_data_moons(dimensions=self.dimension_number, random_seed=42,
                                                      num_points=self.n)
        else:
            x_train, x_test, y_train, y_test = synth_data(dimensions=self.dimension_number, random_seed=42, num_points=int(self.n/2))
        #x_test = x_test[[17,18]]
        #y_test = y_test[[17,18]]
        print("The shape of X train:")
        print(x_train.shape)

        # Plot original dataset:
        plotter1=ClassifierPlotter(x_train, y_train, x_lim=[-3,3], y_lim=[-3,3])
        plotter1.plot_orig_dataset(title="Original TRAINING data")
        plotter2=ClassifierPlotter(x_test, y_test, x_lim=[-3,3], y_lim=[-3,3])
        plotter2.plot_orig_dataset(title="Original TEST data")

        # Train a linear classifier on the data
        if linear:
            f = LinearSVC(dual=False)
            self.threshold=1e-06
        else:
            print("random forest is applied")
            f = RandomForestClassifier(n_estimators=10, random_state=24)
            #f = MLPClassifier(hidden_layer_sizes=(10,10,10), activation='relu', solver='adam', max_iter=1000, random_state=24)
            #f=KNeighborsClassifier(n_neighbors=11)
        f.fit(x_train, y_train)
        self.initial_model = copy.deepcopy(f)

        if linear:
            f.decision_function(x_test)

        # Plot the decision surface
        plotter1.plot_decision_surface(f, title="Original TRAINING data with linear SVC decision boundary")
        plotter2.plot_decision_surface(f, title="Original TEST data with linear SVC decision boundary")

        #  Extract and save the coefficient weights to w_f
        if linear:
         w_f = f.coef_

        # Get accuracies:
        y_train_pred = f.predict(x_train)
        y_test_pred = f.predict(x_test)

        train_accuracy = accuracy_score(y_train, y_train_pred)
        test_accuracy = accuracy_score(y_test, y_test_pred)

        # Display the coefficient weights
        print("--Original model--")
        if linear:
            print("Coefficient weights w_f:", w_f)
        print("Training accuracy:", train_accuracy)
        print("Test accuracy:", test_accuracy)

        model_type = "dec_f" if linear else "pred_proba"

        # 1. FULL INFORMATION
        bestresponse=Fullinformation(x_test, self.strat_features.tolist())
        x_test_shifted1 = bestresponse.algorithm2(self.alpha, f, self.t, self.eps, mod_type=model_type, threshold= self.threshold) #add a small threshhold to make it positive for sure

        #checking if they are orthogonal:
        if linear:
            diff_vectors = x_test_shifted1 - x_test
            angles = np.degrees(np.arctan2(diff_vectors[:, 1], diff_vectors[:, 0]))
            angle_f=np.degrees(np.arctan2(w_f[0][1],w_f[0][0]))
            print(angles)
            print(angle_f)
            print(angles-angle_f)

        ind_failed=bestresponse.get_failed_ind()

        plotter2.plot_decision_surface(f, title="FULL_INFO best responses on linear SVC decision boundary", X_shifted=x_test_shifted1, highlighted_ind_x=ind_failed)

        ## check people who changed:
        x_changes=bestresponse.find_differences()
        costs = bestresponse.get_costs()
        avg_cost = np.sum(costs) / len(x_changes)

        ## new labels by the model:
        y_test_pred_shifted1=f.predict(x_test_shifted1)
        avg_cont_payoff = np.sum(self.t * y_test_pred_shifted1 - costs) / len(y_test_pred_shifted1)

        ## Accuracy after shift:
        test_accuracy_shift1 = accuracy_score(y_test, y_test_pred_shifted1)

        # Social welfare: ground truth labels
        social_welfare=np.sum(y_test == 1) / len(y_test) * 100
        #social_welfare_shift1=np.sum(y_test_shifted1 == 1) / len(y_test_shifted1) * 100

        # User welfare: predicted labels
        user_welfare = np.sum(y_test_pred == 1) / len(y_test_pred) * 100
        user_welfare_shift1=np.sum(y_test_pred_shifted1 == 1) / len(y_test_pred_shifted1) * 100

        print("Accuracy before the shift:", test_accuracy*100, "%")
        print("Social welfare before shift:", social_welfare, "%")
        print("User welfare before shift:", user_welfare, "%")
        print("--")
        print("--1. Full information shift--")
        print("Number of users who changed:", len(x_changes), " in %:", len(x_changes)/len(x_test)*100, "%")
        print("Average cost of change: ", avg_cost)
        print("Accuracy after the shift:", test_accuracy_shift1*100, "%")
        #print("Social welfare after shift:", social_welfare_shift1, "%")
        print("User welfare after shift:", user_welfare_shift1, "%")

        # 2. PARTIAL INFORMATION - LIME
        # Create a LIME explainer
        Lime=PartialInformation(x_train, x_test, self.strat_features)

        x_test_shifted2=Lime.algorithm3(f, 0.4, self.alpha, self.eps, mod_type=model_type)

        plotter2.plot_decision_surface(f, title="PART_INFO best responses with linear SVC decision boundary",
                                       X_shifted=x_test_shifted2)

        x_changes2 = Lime.find_differences()
        costs2 = Lime.get_costs()
        avg_cost2=np.sum(costs2)/len(x_changes2)

        y_test_pred_shifted2 = f.predict(x_test_shifted2)
        user_welfare_shift2 = np.sum(y_test_pred_shifted2 == 1) / len(y_test_pred_shifted2) * 100
        test_accuracy_shift2 = accuracy_score(y_test, y_test_pred_shifted2)

        avg_cont_payoff2=np.sum(self.t*y_test_pred_shifted2-costs2)/len(y_test_pred_shifted2)

        print("--")
        print("--2. Partial information--")
        print("Number of users who changed:", len(x_changes2), " in %:", len(x_changes2)/len(x_test)*100, "%")
        print("Average cost of change: ", avg_cost2)
        print("Accuracy after the shift:", test_accuracy_shift2*100, "%")
        #print("Social welfare after shift:", social_welfare_shift1, "%")
        print("User welfare after shift:", user_welfare_shift2, "%")
        print("Average contestant payoff:", avg_cont_payoff2)


        # 3.1. NO INFORMATION - UTILITY MAXIMIZATION
        sigma = 1.0  # Bandwidth parameter for weigthed sampling

        alg4=NoInformation(x_test, self.strat_features, self.alpha,  self.eps, y_test=y_train, plotting_ind=1)

        x_test_shifted3=alg4.algorithm4_utility(x_train, y_train_pred, sigma, int(self.n/100), self.t, threshold=self.threshold)
        plotter2.plot_decision_surface(f, title="NO_INFO_EST best responses on linear SVC decision boundary", X_shifted=x_test_shifted3)

        x_changes3=alg4.find_differences()
        costs3 = alg4.get_costs()
        avg_cost3 = np.sum(costs3) / len(x_changes3)

        alg4.plot_sample()

        y_test_pred_shifted3=f.predict(x_test_shifted3)
        user_welfare_shift3=np.sum(y_test_pred_shifted3 == 1) / len(y_test_pred_shifted3) * 100
        test_accuracy_shift3 = accuracy_score(y_test, y_test_pred_shifted3)

        avg_cont_payoff3 = np.sum(self.t * y_test_pred_shifted3 - costs3) / len(y_test_pred_shifted3)

        print("--")
        print("--3.1. No information, utility maximalization--")
        print("Number of users who changed:", len(x_changes3), " in %:", len(x_changes3)/len(x_test)*100, "%")
        print("Average cost of change: ", avg_cost3)
        print("Accuracy after the shift:", test_accuracy_shift3*100, "%")
        #print("Social welfare after shift:", social_welfare_shift1, "%")
        print("User welfare after shift:", user_welfare_shift3, "%")
        print("Average contestant payoff:", avg_cont_payoff3)

        # 3.2. NO INFORMATION - IMITATION
        alg4 = NoInformation(x_test, self.strat_features, self.alpha, self.eps)
        x_test_shifted4=alg4.algorithm4_imitation(x_train,y_train_pred, sigma, 50, self.t)
        plotter2.plot_decision_surface(f, title="PART_INFO_IMIT best responses on linear SVC decision boundary", X_shifted=x_test_shifted4)

        x_changes4=alg4.find_differences()
        costs4 = alg4.get_costs()
        avg_cost4 = np.sum(costs4) / len(x_changes4)

        y_test_pred_shifted4=f.predict(x_test_shifted4)
        user_welfare_shift4=np.sum(y_test_pred_shifted4 == 1) / len(y_test_pred_shifted4) * 100
        test_accuracy_shift4 = accuracy_score(y_test, y_test_pred_shifted4)

        avg_cont_payoff4 = np.sum(self.t * y_test_pred_shifted4 - costs4) / len(y_test_pred_shifted4)

        print("--")
        print("--3.2. No information, imitation--")
        print("Number of users who changed:", len(x_changes4), " in %:", len(x_changes4)/len(x_test)*100, "%")
        print("Average cost of change: ", avg_cost4)
        print("Accuracy after the shift:", test_accuracy_shift4*100, "%")
        #print("Social welfare after shift:", social_welfare_shift1, "%")
        print("User welfare after shift:", user_welfare_shift4, "%")
        print("Average contestant payoff:", avg_cont_payoff4)

        # Create table overview:
        index_labels = [
            "Accuracy",
            "User welfare",
            "Social welfare",
            "% of user changes",
            "Average cost of change per change",
            "Average contestant payoff"
        ]
        data = {
            "Original": [test_accuracy*100, user_welfare, social_welfare, None, None, None],
            "FULL_INF": [test_accuracy_shift1*100, user_welfare_shift1, None, len(x_changes)/len(x_test)*100, avg_cost, avg_cont_payoff],
            "PART_INF": [test_accuracy_shift2*100, user_welfare_shift2, None, len(x_changes2)/len(x_test)*100, avg_cost2, avg_cont_payoff2],
            "NO_INF_EST": [test_accuracy_shift3*100, user_welfare_shift3, None, len(x_changes3)/len(x_test)*100, avg_cost3, avg_cont_payoff3],
            "NO_INF_IMIT": [test_accuracy_shift4*100, user_welfare_shift4, None, len(x_changes4)/len(x_test)*100, avg_cost4, avg_cont_payoff4]
        }
        df = pd.DataFrame(data, index=index_labels)

        print(df)

        self.results=df
        df.to_csv(f"outputs/{self.tablename}", index=True, float_format='%.2f')

        # Create a 2x3 grid for the subplots
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        # Flatten axes for easier indexing
        axes = axes.flatten()

        # Define the configurations for the plots
        plots = [
            {"plotter": plotter1, "title": "Original TRAINING data", "data": None, "x_points": None},
            {"plotter": plotter2, "title": "1: Full information TEST data shifted",
             "data": x_test_shifted1, "x_points": ind_failed},
            {"plotter": plotter2, "title": "3.1: No information estimation TEST data shifted",
             "data": x_test_shifted3, "x_points": None},
            {"plotter": plotter2, "title": "Original TEST data", "data": None, "x_points": None},
            {"plotter": plotter2, "title": "2: Partial information TEST data shifted",
             "data": x_test_shifted2, "x_points": None},
            {"plotter": plotter2, "title": "3.2: No information imitation TEST data shifted",
             "data": x_test_shifted4, "x_points": None},
        ]

        # Plot each configuration on the corresponding subplot
        for i, plot_config in enumerate(plots):
            ax = axes[i]  # Get the corresponding subplot axis
            plotter = plot_config["plotter"]  # Determine which plotter to use

            # Call the appropriate function with the axis
            if plot_config["data"] is not None:
                plotter.plot_decision_surface(
                    f,
                    title=plot_config["title"],
                    X_shifted=plot_config["data"],
                    highlighted_ind_x=plot_config["x_points"],
                    ax=ax  # Pass the subplot axis
                )
            else:
                plotter.plot_decision_surface(
                    f,
                    title=plot_config["title"],
                    ax=ax  # Pass the subplot axis
                )

        # Adjust layout for better spacing
        plt.tight_layout()
        plt.subplots_adjust(hspace=0.2)

        # Save the final figure
        plt.savefig(f"outputs/{self.plotname}")

        # Display the plot
        plt.show()

        # save relevant information to self:
        self.x_test=np.copy(x_test)
        self.x_train = np.copy(x_train)
        self.y_test=np.copy(y_test)

    def pop_plotter(self):
        m_list = [self.n * (i*5 / 100) for i in range(1, 19)]
        # 3.1. NO INFORMATION - UTILITY MAXIMIZATION with different sample sizes
        sigma = 1.0  # Bandwidth parameter for weigthed sampling
        y_train_pred=self.initial_model.predict(self.x_train)

        alg4=NoInformation(self.x_test, self.strat_features, self.alpha,  self.eps)
        errors_pop=[]
        for m in m_list:
            x_test_shifted3=alg4.algorithm4_utility(self.x_train, y_train_pred, sigma, int(m), 2*self.t, threshold=self.threshold )
            y_test_pred_shifted3=self.initial_model.predict(x_test_shifted3)
            errors_pop.append(100-accuracy_score(self.y_test, y_test_pred_shifted3)*100)
        plt.plot(m_list, errors_pop, label="3.1. No information")

        # fully informed case:
        error_full = 100-self.results["FULL_INFO"]["Accuracy"]
        error_non = 100-self.results["Original"]["Accuracy"]
        error_part=100-self.results["PART_INFO"]["Accuracy"]
        plt.axhline(y=error_non, color='red', linestyle='--', label="Non-strategic")
        plt.axhline(y=error_full, color='blue', linestyle='-', label="FULL_INFO")
        plt.axhline(y=error_part, color='orange', linestyle=':',  label="PART_INFO")

        plt.fill_between(
            m_list,  # x values (indices of accuracies_pop)
            error_full,  # Lower boundary (the plotted accuracies)
            errors_pop,  # Upper boundary (the horizontal line)
            #where=(np.array(errors_pop) >= error_full),  # Only fill where condition is true
            facecolor='#DBDBDB',  # Fill color
            alpha=0.3,  # Transparency
            hatch='///',  # Striped pattern
            edgecolor = 'grey',
            linewidth = 0.0
        )

        plt.xlabel("m (sample size)")
        plt.ylabel("Error [%]")
        plt.title("Errors in different information scenarios")
        plt.legend()

        plt.xlim(m_list[0],m_list[(len(m_list)-1)])

        plt.savefig(f"outputs/moons_lin_pop.pdf")

        plt.show()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dimension_number", type=int, default=2,
                    help="Number of dimensions for synthetic data.")
    parser.add_argument("--n", type=int, default=400,
                        help="Number of datapoints in total for synthetic data. 10% test, 90% training split.")
    parser.add_argument("--t", type=float, default=2,
                        help="Controls the budget for changes. The higher t, the higher costs users are willing to pay.")
    parser.add_argument("--eps", type=float, default=2,
                        help="The weight of the quadratic element in the mixed cost function.")
    parser.add_argument("--moons", action="store_true",
                        help="If set, the moons dataset is used. Otherwise, synthetic data has Gaussian clusters.")
    parser.add_argument("--linear", action="store_true",
                        help="If set, a linear SVC classifier is used. Otherwise, a kernelized SVC is applied.")

    args = parser.parse_args()

    processor = main_class(
        dimension_number=args.dimension_number,
        n=args.n,
        t=args.t,
        eps=args.eps,
    )

    processor.info_comparison(moons=args.moons, linear=args.linear)
    processor.pop_plotter()
