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
    def __init__(self, n: int, t: float, eps : float) -> None:
        self.t = t  #/5
        self.eps = eps
        self.feature_names = ["Feature1", "Feature2"]
        self.n=n
        self.threshold=0.5+1e-03

    def info_comparison(self, moons: bool, loan: bool, model_type: str):
        if moons:
            x_train, x_test, y_train, y_test = synth_data_moons(dimensions=2, random_seed=42,
                                                      num_points=self.n)
            self.strat_features = np.array([0, 1])
            self.alpha = 0.5 * np.array([0.5, 0.5]).reshape(2,1)
        elif loan==False:
            x_train, x_test, y_train, y_test = synth_data(dimensions=2, random_seed=42, num_points=int(self.n/2))
            self.strat_features = np.array([0, 1])
            self.alpha = np.array([0.5, 0.5]).reshape(2,1)
        else:
            x_train, x_test, y_train, y_test = loan_data(train_val=True)
            self.strat_features = np.array([0, 1, 2, 3, 4, 5])
            self.alpha = 0.5 * np.array([0.5, 0.5, 1.5, -2.5, -0.5, 0.5]).reshape(6, 1)


        print("The shape of X train:")
        print(x_train.shape)

        print("The shape of X test:")
        print(x_test.shape)

        # Plot original dataset:
        if not loan:
            plotter1=ClassifierPlotter(x_train, y_train, x_lim=[-3,3], y_lim=[-3,3])
            plotter1.plot_orig_dataset(title="Original TRAINING data")
            plotter2=ClassifierPlotter(x_test, y_test, x_lim=[-3,3], y_lim=[-3,3])
            plotter2.plot_orig_dataset(title="Original TEST data")

        # Train a linear classifier on the data
        if model_type=="rnf":
            print("random forest is applied")
            f = RandomForestClassifier(n_estimators=10, random_state=24)
        elif model_type=="nn":
            print("Neural network is applied")
            f = MLPClassifier(hidden_layer_sizes=(10,10,10,10), activation='relu', solver='adam', max_iter=1000, random_state=24)
        elif model_type=="knn":
            print("K-nearest neighbors classifier is applied")
            f=KNeighborsClassifier(n_neighbors=11)
        else:
            print("Linear model is applied")
            f = LinearSVC(C=0.01, penalty='l2', random_state=42)
            self.threshold = 1e-06

        f.fit(x_train, y_train)
        self.initial_model = copy.deepcopy(f)

        if model_type=="linear":
            f.decision_function(x_test)

        # Plot the decision surface
        if not loan:
            plotter1.plot_decision_surface(f, title=f"Original TRAINING data with {args.model_type.upper()} decision boundary")
            plotter2.plot_decision_surface(f, title=f"Original TEST data with {args.model_type.upper()} decision boundary")

        #  Extract and save the coefficient weights to w_f
        if model_type=="linear":
         w_f = f.coef_

        # Get accuracies:
        y_train_pred = f.predict(x_train)
        y_test_pred = f.predict(x_test)

        train_accuracy = accuracy_score(y_train, y_train_pred)
        test_accuracy = accuracy_score(y_test, y_test_pred)

        # Display the coefficient weights
        print("--Original model--")
        if model_type=="linear":
            print("Coefficient weights w_f:", w_f)
        print("Training accuracy:", train_accuracy)
        print("Test accuracy:", test_accuracy)

        model_type = "dec_f" if model_type=="linear" else "pred_proba"

        # 1. FULL INFORMATION
        bestresponse=Fullinformation(x_test, self.strat_features.tolist())
        x_test_shifted1 = bestresponse.algorithm2(self.alpha, f, self.t, self.eps, mod_type=model_type, threshold= self.threshold) #add a small threshhold to make it positive for sure

        #checking if they are orthogonal:
        if model_type=="linear":
            diff_vectors = x_test_shifted1 - x_test
            angles = np.degrees(np.arctan2(diff_vectors[:, 1], diff_vectors[:, 0]))
            angle_f=np.degrees(np.arctan2(w_f[0][1],w_f[0][0]))

        ind_failed=bestresponse.get_failed_ind()

        if not loan:
            plotter2.plot_decision_surface(f, title=f"FULL_INFO best responses on {args.model_type.upper()} decision boundary", X_shifted=x_test_shifted1, highlighted_ind_x=ind_failed)

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

        if not loan:
            plotter2.plot_decision_surface(f, title=f"PART_INFO best responses with {args.model_type.upper()} decision boundary",
                                       X_shifted=x_test_shifted2)

        x_changes2 = Lime.find_differences()
        costs2 = Lime.get_costs()
        avg_cost2=np.sum(costs2)/len(x_changes2)

        y_test_pred_shifted2 = f.predict(x_test_shifted2)
        user_welfare_shift2 = np.sum(y_test_pred_shifted2 == 1) / len(y_test_pred_shifted2) * 100
        test_accuracy_shift2 = accuracy_score(y_test, y_test_pred_shifted2)

        avg_cont_payoff2=np.sum(self.t*y_test_pred_shifted2-costs2)/len(y_test_pred_shifted2)

        # Get disagreement set:
        y_test_pred_shifted2_user= Lime.est_pred_after()
        disagreement_set_2=np.where(y_test_pred_shifted2_user != y_test_pred_shifted2)
        size_of_dis_set_2 = len(disagreement_set_2[0])

        print("--")
        print("--2. Partial information--")
        print("Number of users who changed:", len(x_changes2), " in %:", len(x_changes2)/len(x_test)*100, "%")
        print("Average cost of change: ", avg_cost2)
        print("Accuracy after the shift:", test_accuracy_shift2*100, "%")
        #print("Social welfare after shift:", social_welfare_shift1, "%")
        print("User welfare after shift:", user_welfare_shift2, "%")
        print("Average contestant payoff:", avg_cont_payoff2)
        print("The size of disagreement set:", size_of_dis_set_2, "%:", size_of_dis_set_2/len(y_test_pred_shifted2)*100)


        # 3.1. NO INFORMATION - UTILITY MAXIMIZATION
        sigma = 1.0  # Bandwidth parameter for weigthed sampling

        alg4=NoInformation(x_test, self.strat_features, self.alpha,  self.eps, y_test=y_train, plotting_ind=43)

        x_test_shifted3=alg4.algorithm4_utility(x_train, y_train_pred, sigma, int(self.n/100), self.t)

        if not loan:
            plotter2.plot_decision_surface(f, title=f"NO_INFO_EST best responses on {args.model_type.upper()} decision boundary", X_shifted=x_test_shifted3)

        x_changes3=alg4.find_differences()
        costs3 = alg4.get_costs()
        avg_cost3 = np.sum(costs3) / len(x_changes3)

        if not loan:
            alg4.plot_sample()
            alg4.plot_sample(estimated=False, model_true=f)

        y_test_pred_shifted3=f.predict(x_test_shifted3)
        user_welfare_shift3=np.sum(y_test_pred_shifted3 == 1) / len(y_test_pred_shifted3) * 100
        test_accuracy_shift3 = accuracy_score(y_test, y_test_pred_shifted3)

        avg_cont_payoff3 = np.sum(self.t * y_test_pred_shifted3 - costs3) / len(y_test_pred_shifted3)

        # Get disagreement set:
        y_test_pred_shifted3_user= alg4.est_pred_after()
        disagreement_set_3=np.where(y_test_pred_shifted3_user != y_test_pred_shifted3)
        size_of_dis_set_3 = len(disagreement_set_3[0])

        print("--")
        print("--3.1. No information, utility maximalization--")
        print("Number of users who changed:", len(x_changes3), " in %:", len(x_changes3)/len(x_test)*100, "%")
        print("Average cost of change: ", avg_cost3)
        print("Accuracy after the shift:", test_accuracy_shift3*100, "%")
        #print("Social welfare after shift:", social_welfare_shift1, "%")
        print("User welfare after shift:", user_welfare_shift3, "%")
        print("Average contestant payoff:", avg_cont_payoff3)
        print("The size of disagreement set:", size_of_dis_set_3, "%:", size_of_dis_set_3/len(y_test_pred_shifted3)*100)

        # 3.2. NO INFORMATION - IMITATION
        alg4 = NoInformation(x_test, self.strat_features, self.alpha, self.eps, plotting_ind=1)
        x_test_shifted4=alg4.algorithm4_imitation(x_train,y_train_pred, sigma, int(self.n/100), 2*self.t)

        if not loan:
            plotter2.plot_decision_surface(f, title=f"NO_INFO_IMIT best responses on {args.model_type.upper()} decision boundary", X_shifted=x_test_shifted4)
            #alg4.plot_sample()

        x_changes4=alg4.find_differences()
        costs4 = alg4.get_costs()
        avg_cost4 = np.sum(costs4) / len(x_changes4)

        y_test_pred_shifted4=f.predict(x_test_shifted4)
        user_welfare_shift4=np.sum(y_test_pred_shifted4 == 1) / len(y_test_pred_shifted4) * 100
        test_accuracy_shift4 = accuracy_score(y_test, y_test_pred_shifted4)

        avg_cont_payoff4 = np.sum(self.t * y_test_pred_shifted4 - costs4) / len(y_test_pred_shifted4)


        print("--")
        print("--3.2. No information, imitation--")
        print("Number of users who changed:", len(x_changes4), ", in %:", len(x_changes4)/len(x_test)*100, "%")
        print("Average cost of change: ", avg_cost4)
        print("Accuracy after the shift:", test_accuracy_shift4*100, "%")
        #print("Social welfare after shift:", social_welfare_shift1, "%")
        print("User welfare after shift:", user_welfare_shift4, "%")
        print("Average contestant payoff:", avg_cont_payoff4)

        # Get the enlargement set:
        all_match = (y_test_pred_shifted1 == y_test_pred_shifted2) & (y_test_pred_shifted1 == y_test_pred_shifted3) & (
                    y_test_pred_shifted1 == y_test_pred_shifted4)

        # Find indices where not all match (i.e., where the condition is False)
        indices_not_matching = np.where(~all_match)[0]

        # Check it also for pairs with full info:
        sc2=100 * len(np.where(~(y_test_pred_shifted1 == y_test_pred_shifted2))[0]) / len(y_test_pred_shifted1)
        sc3 = 100 * len(np.where(~(y_test_pred_shifted1 == y_test_pred_shifted3))[0]) / len(y_test_pred_shifted1)
        sc4 = 100 * len(np.where(~(y_test_pred_shifted1 == y_test_pred_shifted4))[0]) / len(y_test_pred_shifted1)

        print("--")
        print("The size of the enlargement set:", len(indices_not_matching), "in %:", 100*len(indices_not_matching)/len(y_test_pred_shifted4))
        print("Pairwise enlargement sets: ",
              "Full and LIME", sc2,
              "Full and PART_EST", sc3,
              "Full and PART_IMIT", sc4)
        print("2* Error(f, f) in %:", 200 * (1 - test_accuracy_shift1))

        if sc2>200 * (1 - test_accuracy_shift1):
            print("Sufficient condition is true for Full Info & LIME, POP should be positive!")
        if sc3>200 * (1 - test_accuracy_shift1):
            print("Sufficient condition is true for Full Info & NO_INFO_EST, POP should be positive!")
        if sc4>200 * (1 - test_accuracy_shift1):
            print("Sufficient condition is true for Full Info & NO_INFO_IMIT, POP should be positive!")



        # Create table overview:
        index_labels = [
            "Accuracy",
            "User welfare",
            "Social welfare",
            "% of user changes",
            "Average cost of change per change",
            "Average contestant payoff",
            "Size of disagreement set %"
        ]
        data = {
            "Original": [test_accuracy*100, user_welfare, social_welfare, None, None, None, None],
            "FULL\_INFO": [test_accuracy_shift1*100, user_welfare_shift1, None, len(x_changes)/len(x_test)*100, avg_cost, avg_cont_payoff, None],
            "PART\_INFO": [test_accuracy_shift2*100, user_welfare_shift2, None, len(x_changes2)/len(x_test)*100, avg_cost2, avg_cont_payoff2, size_of_dis_set_2/len(y_test_pred_shifted2)*100],
            "NO\_INFO\_EST": [test_accuracy_shift3*100, user_welfare_shift3, None, len(x_changes3)/len(x_test)*100, avg_cost3, avg_cont_payoff3, size_of_dis_set_3/len(y_test_pred_shifted3)*100],
            "NO\_INFO\_IMIT": [test_accuracy_shift4*100, user_welfare_shift4, None, len(x_changes4)/len(x_test)*100, avg_cost4, avg_cont_payoff4, None]
        }
        df = pd.DataFrame(data, index=index_labels)

        df = df.drop(df.index[2]) # drop social welfare

        print(df)

        self.results=df
        df.to_csv(f"outputs/{dataset_name}_{tablename}.csv", index=True, float_format='%.2f') #variations on cost budget/

        if not loan:
            # Create a 2x3 grid for the subplots
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))

            # Flatten axes for easier indexing
            axes = axes.flatten()

            # Define the configurations for the plots
            plots = [
                {"plotter": plotter1, "title": f"Original TRAINING data with trained {args.model_type.upper()} decision boundary", "data": None, "x_points": None},
                {"plotter": plotter2, "title": "1: FULL_INFO best responses",
                 "data": x_test_shifted1, "x_points": ind_failed},
                {"plotter": plotter2, "title": "3.1: NO_INFO_EST best responses",
                 "data": x_test_shifted3, "x_points": None},
                {"plotter": plotter2, "title": "Original TEST data", "data": None, "x_points": None},
                {"plotter": plotter2, "title": "2: PART_INFO best responses",
                 "data": x_test_shifted2, "x_points": None},
                {"plotter": plotter2, "title": "NO_INFO_IMIT best responses",
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
            plt.savefig(f"outputs/{dataset_name}_{plotname1}.pdf")

            # Display the plot
            plt.show()

        # save relevant information to self:
        self.x_test=np.copy(x_test)
        self.x_train = np.copy(x_train)
        self.y_test=np.copy(y_test)

    def pop_plotter(self, loan: bool):
        if loan:
            m_list = [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
        else:
            m_list = [4, 8, 16, 32, 64, 128, 256, 512, 1024]
        # 3.1. NO INFORMATION -
        sigma = 1.0  # Bandwidth parameter for weigthed sampling
        y_train_pred=self.initial_model.predict(self.x_train)

        alg4=NoInformation(self.x_test, self.strat_features, self.alpha,  self.eps)
        errors_pop=[]
        errors_pop2 = []
        for m in m_list:
            # UTILITY MAXIMIZATION with different sample sizes
            x_test_shifted3=alg4.algorithm4_utility(self.x_train, y_train_pred, sigma, int(m), 2*self.t)
            y_test_pred_shifted3=self.initial_model.predict(x_test_shifted3)
            errors_pop.append(100-accuracy_score(self.y_test, y_test_pred_shifted3)*100)

            # IMITATION with different sample sizes
            x_test_shifted3 = alg4.algorithm4_imitation(self.x_train, y_train_pred, sigma, int(m), 2 * self.t)
            y_test_pred_shifted3 = self.initial_model.predict(x_test_shifted3)
            errors_pop2.append(100 - accuracy_score(self.y_test, y_test_pred_shifted3) * 100)
        plt.plot(m_list, errors_pop, label="NO_INFO_EST")
        plt.plot(m_list, errors_pop2, label="NO_INFO_IMIT")

        # fully informed case:
        error_full = 100-self.results["FULL\_INFO"]["Accuracy"]
        error_non = 100-self.results["Original"]["Accuracy"]
        error_part=100-self.results["PART\_INFO"]["Accuracy"]
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

        #Adding POP arrow annotation
        m_value = 8
        plt.annotate(
            '',
            xy=(m_value, errors_pop[1]),  # Arrowhead location
            xytext=(m_value, error_full),  # Arrow tail location
            arrowprops=dict(arrowstyle='<->', color='black', linewidth=1.5)  # Double-headed vertical arrow
        )

        text = "+POP" if errors_pop[1]-error_full>0 else "-POP"

        # Optionally, label the arrow
        plt.text(m_value + 0.5, (error_full +  errors_pop[1]) / 2, text,
                 verticalalignment='center', fontsize=12)

        plt.xlabel("m (sample size)")
        plt.ylabel("Error [%]")
        plt.title(f"Errors in different information scenarios for {args.model_type.upper()} model")
        plt.legend()

        plt.xscale("log")

        plt.xlim(m_list[0], m_list[-1])
        if loan:
            plt.ylim(15,50)
        else:
            plt.ylim(5, 50)

        plt.savefig(f"outputs/{dataset_name}_{plotname2}.pdf")

        plt.show()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=1138,
                        help="Number of datapoints in total for synthetic data. 10% test, 90% training split.")
    parser.add_argument("--t", type=float, default=0.2,
                        help="Controls the budget for changes. The higher t, the higher costs users are willing to pay.")
    parser.add_argument("--eps", type=float, default=0.2,
                        help="The weight of the quadratic element in the mixed cost function.")
    parser.add_argument("--moons", action="store_true",
                        help="If set, the moons dataset is used. Otherwise, synthetic data has Gaussian clusters.")
    parser.add_argument("--model_type", type=str, default="linear", choices=["linear", "nn", "knn", "rnf"],
        help="Choose model type. Default: linear. Other options: nn, knn, and rnf."
    )
    parser.add_argument("--loan", action="store_true",
                        help="If set, the loans dataset is used. Otherwise, a synthetic dataset is used.")

    args = parser.parse_args()

    # Set plotnames
    if args.model_type == "nn":
        plotname1 = "nn"
        plotname2 = "nn_pop"
        tablename = "nn"
    elif args.model_type == "knn":
        plotname1 = "knn"
        plotname2 = "knn_pop"
        tablename = "knn"
    elif args.model_type == "rnf":
        plotname1 = "rnf"
        plotname2 = "rnf_pop"
        tablename = "rnf"
    else: # default is linear model
        plotname1 = "lin"
        plotname2 = "lin_pop"
        tablename = "lin"
    dataset_name= "loan" if args.loan else "moons" if args.moons else "circ"

    processor = main_class(
        n=args.n,
        t=args.t,
        eps=args.eps,
    )

    processor.info_comparison(moons=args.moons, model_type=args.model_type, loan=args.loan)
    processor.pop_plotter(loan=args.loan)
