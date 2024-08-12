import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from best_responses import *
from lime.lime_tabular import LimeTabularExplainer
from weightedsampler import *
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn import preprocessing
from plotting import plot_orig_dataset


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
plot_orig_dataset(x_train, y_train, title="Original TRAINING data")
plot_orig_dataset(x_test, y_test, title="Original TEST data")

# Train a linear classifier on the data
f = LinearSVC(dual=False)
f.fit(x_train, y_train)

#  Extract and save the coefficient weights to w_f
w_f = f.coef_

# Get accuracies:
y_train_pred = f.predict(x_train)
y_test_pred = f.predict(x_test)

train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

# Display the coefficient weights
print("Coefficient weights w_f:", w_f)
print("Training accuracy:", train_accuracy)
print("Test accuracy:", test_accuracy)

# 1. FULL INFORMATION
bestresponse=BestResponse(x_test, strat_features)
x_test_shifted1 = bestresponse.algorithm2(alpha, f, t, eps, mod_type="dec_f", treshold=0.5)

## check people who changed:
x_changes=bestresponse.find_differences()
costs2 = bestresponse.get_costs()
print(costs2)

## new labels by the model and ground truth:
y_test_pred_shifted1=f.predict(x_test_shifted1)
y_test_shifted1=np.sign(x_test_shifted1)

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
print("Number of users who changed:", len(x_changes))
print("Accuracy after the shift:", test_accuracy_shift1*100, "%")
print("Social welfare after shift:", social_welfare_shift1, "%")
print("User welfare after shift:", user_welfare_shift1, "%")


# 3.1. NO INFORMATION - UTILITY MAXIMIZATION
x = x_train[0]  # Given datapoint
sigma = 1.0  # Bandwidth parameter

y_pred_estimation=bestresponse.algorithm4(x_train, y_train, sigma, 50)
print(y_pred_estimation)

alg4=algorithm4(x_test, strat_features)

x_test_shifted2=alg4.sample_predict_shift_utility(x_train, y_train, sigma, 50, alpha, t, eps, treshold=0.5 )

costs_31=alg4.get_costs()


print()
print(x_test_shifted2)
print(x_test)

# 3.2. NO INFORMATION - IMITATION
x_test_shifted3=alg4.sample_predict_shift_imitation(x_train,y_train, sigma, 50, 0.1, alpha, eps)

print(x_test_shifted3)







# 2. PARTIAL INFORMATION

f = LinearSVC(dual=False)
f.fit(x_train, y_train)

f2=MLPClassifier(hidden_layer_sizes=(50,10))
f2.fit(x_train, y_train)

# Create a LIME explainer
explainer = LimeTabularExplainer(
    training_data=x_train,
    feature_names=['feature_1'],
    class_names=['class_0', 'class_1'],
    mode='classification'
)

def predict_proba_linSVC(X):
    decision = f.decision_function(X)
    expit = lambda x: 1 / (1 + np.exp(-x))
    proba = np.apply_along_axis(expit, 0, decision)
    return np.column_stack([1-proba, proba]) if proba.ndim == 1 else proba

haha=predict_proba_linSVC(x_train[0].reshape(-1, 1))
hihi=f.decision_function(x_train[0].reshape(-1, 1))


# Explain the first instance
exp = explainer.explain_instance(
    data_row=x_train[0],
    predict_fn=predict_proba_linSVC
)


