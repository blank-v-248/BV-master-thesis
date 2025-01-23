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

import warnings
from sklearn.exceptions import DataConversionWarning
import copy

# Suppress the specific warning
warnings.filterwarnings(action='ignore', category=DataConversionWarning)

class main_class:
    def __init__(self, dimension_number: int, plotname: str, tablename: str) -> None:
        self.dimension_number = dimension_number
        self.plotname = plotname
        self.tablename = tablename
        self.strat_features = np.array([0, 1])
        self.alpha = np.array([1, 1]).reshape(2, 1)
        self.t = 2
        self.eps = 0.8
        self.feature_names = ["Feature1", "Feature2"]
        self.n=200

    def info_comparison(self):

        x_train, x_test, y_train, y_test = synth_data(dimensions=self.dimension_number, random_seed=42, num_points=int(self.n/2))
        x_test=x_test[10:19]
        y_test = y_test[10:19]

        print("The shape of X train:")
        print(x_train.shape)

        # Plot original dataset:
        plotter1=ClassifierPlotter(x_train, y_train, x_lim=[-3,3], y_lim=[-3,3])
        plotter1.plot_orig_dataset(title="Original TRAINING data")
        plotter2=ClassifierPlotter(x_test, y_test, x_lim=[-3,3], y_lim=[-3,3])
        plotter2.plot_orig_dataset(title="Original TEST data")

        # Train a linear classifier on the data
        f = LinearSVC(dual=False)
        f.fit(x_train, y_train)
        self.initial_model = copy.deepcopy(f)

        f.decision_function(x_test)

        # Plot the decision surface
        plotter1.plot_decision_surface(f, title="Original TRAINING data with linear SVC decision boundary")
        plotter2.plot_decision_surface(f, title="Original TEST data with linear SVC decision boundary")

        #  Extract and save the coefficient weights to w_f
        w_f = f.coef_

        # Get accuracies:
        y_train_pred = f.predict(x_train)
        y_test_pred = f.predict(x_test)

        train_accuracy = accuracy_score(y_train, y_train_pred)
        test_accuracy = accuracy_score(y_test, y_test_pred)

        # Display the coefficient weights
        print("--Original model--")
        print("Coefficient weights w_f:", w_f)
        print("Training accuracy:", train_accuracy)
        print("Test accuracy:", test_accuracy)

        # 1. FULL INFORMATION
        bestresponse=Fullinformation(x_test, self.strat_features.tolist())
        x_test_shifted1 = bestresponse.algorithm2(self.alpha, f, self.t, self.eps, mod_type="dec_f", threshold=0)

        #checking if they are orthogonal:
        diff_vectors = x_test_shifted1 - x_test
        angles = np.degrees(np.arctan2(diff_vectors[:, 1], diff_vectors[:, 0]))
        angle_f=np.degrees(np.arctan2(w_f[0][1],w_f[0][0]))
        print(angles)
        print(angle_f)
        print(angles-angle_f)

        plotter2.plot_decision_surface(f, title="1: Full information TEST data with linear SVC decision boundary", X_shifted=x_test_shifted1)

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

        x_test_shifted2=Lime.algorithm3(f, 0.4, self.alpha, self.eps)

        plotter2.plot_decision_surface(f, title="2: Partial information TEST data with linear SVC decision boundary",
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

        alg4=NoInformation(x_test, self.strat_features, self.alpha,  self.eps)

        x_test_shifted3=alg4.algorithm4_utility(x_train, y_train_pred, sigma, 50, self.t, threshold=0.5 )
        plotter2.plot_decision_surface(f, title="3.1. No information estimation TEST data with linear SVC decision boundary", X_shifted=x_test_shifted3)

        x_changes3=alg4.find_differences()
        costs3 = alg4.get_costs()
        avg_cost3 = np.sum(costs3) / len(x_changes3)

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
        plotter2.plot_decision_surface(f, title="3.2. No information imitation TEST data with linear SVC decision boundary", X_shifted=x_test_shifted4)

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
            "1": [test_accuracy_shift1*100, user_welfare_shift1, None, len(x_changes)/len(x_test)*100, avg_cost, avg_cont_payoff],
            "2": [test_accuracy_shift2*100, user_welfare_shift2, None, len(x_changes2)/len(x_test)*100, avg_cost2, avg_cont_payoff2],
            "3.1.": [test_accuracy_shift3*100, user_welfare_shift3, None, len(x_changes3)/len(x_test)*100, avg_cost3, avg_cont_payoff3],
            "3.2.": [test_accuracy_shift4*100, user_welfare_shift4, None, len(x_changes4)/len(x_test)*100, avg_cost4, avg_cont_payoff4]
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
            {"plotter": plotter1, "title": "Original TRAINING data", "data": None},
            {"plotter": plotter2, "title": "1: Full information TEST data shifted",
             "data": x_test_shifted1},
            {"plotter": plotter2, "title": "3.1: No information estimation TEST data shifted",
             "data": x_test_shifted3},
            {"plotter": plotter2, "title": "Original TEST data", "data": None},
            {"plotter": plotter2, "title": "2: Partial information TEST data shifted",
             "data": x_test_shifted2},
            {"plotter": plotter2, "title": "3.2: No information imitation TEST data shifted",
             "data": x_test_shifted4},
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
        self.y_train = np.copy(y_train)

    def pop_plotter(self):
        m_list = [self.n * (i*5 / 100) for i in range(1, 11)]
        # 3.1. NO INFORMATION - UTILITY MAXIMIZATION with different sample sizes
        sigma = 1.0  # Bandwidth parameter for weigthed sampling
        y_train_pred=self.initial_model.predict(self.x_train)

        alg4=NoInformation(self.x_test, self.strat_features, self.alpha,  self.eps)
        accuracies_pop=[]
        for m in m_list:
            x_test_shifted3=alg4.algorithm4_utility(self.x_train, y_train_pred, sigma, int(m), 2*self.t, threshold=0.5 )
            y_test_pred_shifted3=self.initial_model.predict(x_test_shifted3)
            accuracies_pop.append( accuracy_score(self.y_test, y_test_pred_shifted3))
        plt.plot(accuracies_pop)
        plt.show()




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dimension_number", type=int, default=2,
                    help="Number of dimensions for synthetic data.")
    parser.add_argument("--plotname", type=str, default="plot.png",
                        help="Plot name for 6 subplots")
    parser.add_argument("--tablename", type=str, default="table.csv",
                        help="Table name for performance metrics")



    args = parser.parse_args()

    processor = main_class(
        dimension_number=args.dimension_number,
        plotname=args.plotname,
        tablename=args.tablename

    )

    processor.info_comparison()
    processor.pop_plotter()
