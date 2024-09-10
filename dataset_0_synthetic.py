import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score


import best_responses
from best_responses import *


from weightedsampler import *
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn import preprocessing
from plotting import *


# Generate 1D normally distributed data
mean = 0  # Expected value (mean)
variance = 1  # Variance
size = 100  # Number of data points
X = np.random.normal(mean, np.sqrt(variance), size).reshape(-1, 1)
y = np.sign(X) # Assign labels based on the sign of x

# Generate 2D cluster data
num_points = 100
np.random.seed(23)

# Cluster 1 centered around (12, 180)
cluster1_x1 = np.random.normal(12, 20, num_points)
cluster1_x2 = np.random.normal(180, 90, num_points)
cluster1_y = np.ones(num_points)  # Label 1

# Cluster 2 centered around (55, 30)
cluster2_x1 = np.random.normal(55, 20, num_points)
cluster2_x2 = np.random.normal(30, 90, num_points)
cluster2_y = np.zeros(num_points)  # Label 0

# Concatenate data from both clusters
x1 = np.concatenate([cluster1_x1, cluster2_x1])
x2 = np.concatenate([cluster1_x2, cluster2_x2])
y = np.concatenate([cluster1_y, cluster2_y])

X = np.column_stack((x1, x2))
std_scaler = preprocessing.StandardScaler()
X = std_scaler.fit_transform(X)

# Define parameters
strat_features=np.array([0, 1])
alpha=np.array([1, 1]).reshape(2,1)
t=2
eps=1
feature_names=["Feature1", "Feature2"]

# Display the first 10 elements to verify
print("First 10 elements of X:", X[:10])
print("First 10 elements of y:", y[:10])

# Split the data into train and test sets with an 80-20% ratio
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
print(x_train.shape)

# Plot original dataset:
plotter1=ClassifierPlotter(x_train, y_train)
#plotter1.plot_orig_dataset(title="Original TRAINING data")
plotter2=ClassifierPlotter(x_test, y_test)
#plotter2.plot_orig_dataset(title="Original TEST data")

# Train a linear classifier on the data
f = LinearSVC(dual=False)
f.fit(x_train, y_train)

f.decision_function(x_test)

# Plot the decision surface
#plotter1.plot_decision_surface(f, title="Original TRAINING data with linear SVC decision boundary")
#plotter2.plot_decision_surface(f, title="Original TEST data with linear SVC decision boundary")

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
bestresponse=BestResponse(x_test, strat_features)
x_test_shifted1 = bestresponse.algorithm2(alpha, f, t, eps, mod_type="dec_f", treshold=0)

plotter2.plot_decision_surface(f, title="Shifted TEST data with linear SVC decision boundary", X_shifted=x_test_shifted1)

## check people who changed:
x_changes=bestresponse.find_differences()
costs2 = bestresponse.get_costs()

## new labels by the model and ground truth:
y_test_pred_shifted1=f.predict(x_test_shifted1)
y_test_shifted1=np.sign(x_test_shifted1) # TO-DO: make ground truth function

## Accuracy after shift:
test_accuracy_shift1 = accuracy_score(y_test, y_test_pred_shifted1)

social_welfare=np.sum(y_test == 1) / len(y_test) * 100
social_welfare_shift1=np.sum(y_test_shifted1 == 1) / len(y_test_shifted1) * 100

user_welfare = np.sum(y_test_pred == 1) / len(y_test_pred) * 100
user_welfare_shift1=np.sum(y_test_pred_shifted1 == 1) / len(y_test_pred_shifted1) * 100

print("Accuracy before the shift:", test_accuracy*100, "%")
print("Social welfare before shift:", social_welfare, "%")
print("User welfare before shift:", user_welfare, "%")
print("--")
print("--1. Full information shift--")
print("Number of users who changed:", len(x_changes))
print("Accuracy after the shift:", test_accuracy_shift1*100, "%")
#print("Social welfare after shift:", social_welfare_shift1, "%")
print("User welfare after shift:", user_welfare_shift1, "%")


# 3.1. NO INFORMATION - UTILITY MAXIMIZATION
x = x_train[0]  # Given datapoint
sigma = 1.0  # Bandwidth parameter

alg4=algorithm4(x_test, strat_features)

x_test_shifted2=alg4.sample_predict_shift_utility(x_train, y_train, sigma, 50, alpha, t, eps, treshold=0.5 )
plotter2.plot_decision_surface(f, title="3 TEST data with linear SVC decision boundary", X_shifted=x_test_shifted2)

x_changes2=alg4.find_differences()
costs2 = alg4.get_costs()

y_test_pred_shifted2=f.predict(x_test_shifted2)
user_welfare_shift2=np.sum(y_test_pred_shifted2 == 1) / len(y_test_pred_shifted2) * 100
test_accuracy_shift2 = accuracy_score(y_test, y_test_pred_shifted2)

print("--")
print("--3.1. No information, utility maximalization--")
print("Number of users who changed:", len(x_changes2))
print("Accuracy after the shift:", test_accuracy_shift2*100, "%")
#print("Social welfare after shift:", social_welfare_shift1, "%")
print("User welfare after shift:", user_welfare_shift2, "%")



# 3.2. NO INFORMATION - IMITATION
x_test_shifted3=alg4.sample_predict_shift_imitation(x_train,y_train, sigma, 50, 0.1, alpha, eps)
plotter2.plot_decision_surface(f, title="Shifted3 TEST data with linear SVC decision boundary", X_shifted=x_test_shifted3)

x_changes3=alg4.find_differences()
costs3 = alg4.get_costs()

y_test_pred_shifted3=f.predict(x_test_shifted3)
user_welfare_shift3=np.sum(y_test_pred_shifted3 == 1) / len(y_test_pred_shifted3) * 100
test_accuracy_shift3 = accuracy_score(y_test, y_test_pred_shifted3)

print("--")
print("--3.2. No information, imitation--")
print("Number of users who changed:", len(x_changes3))
print("Accuracy after the shift:", test_accuracy_shift3*100, "%")
#print("Social welfare after shift:", social_welfare_shift1, "%")
print("User welfare after shift:", user_welfare_shift3, "%")

# 2. PARTIAL INFORMATION - LIME - in work

f = LinearSVC(dual=False)
f.fit(x_train, y_train)

f2=MLPClassifier(hidden_layer_sizes=(50,10))
f2.fit(x_train, y_train)

# Create a LIME explainer
Lime=best_responses.LimeFeatureChanger(x_train, x_test[3:4, ], strat_features)

exp=Lime.explainer(f, 0.8, alpha, eps)
print(exp)









