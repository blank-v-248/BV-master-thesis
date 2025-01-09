
def synth_data(dimensions=2, random_seed=24):
    if dimensions==2:
        # Generate 2D cluster data
        num_points = 100
        np.random.seed(random_seed)

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
        strat_features = np.array([0, 1])
        alpha = np.array([1, 1]).reshape(2, 1)
        t = 2
        eps = 1
        feature_names = ["Feature1", "Feature2"]

        # Split the data into train and test sets with an 80-20% ratio
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=random_seed)
    else:
        raiseError("Please choose a feasible number of dimensions.")

import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

def synth_data(dimensions=2, num_points=100, random_seed=24):
    if dimensions != 2:
        raise ValueError("Please choose a feasible number of dimensions (currently only 2 is supported).")
    else:
        # Set the random seed for reproducibility
        np.random.seed(random_seed)

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

        # Combine the features into a single dataset
        X = np.column_stack((x1, x2))

        # Standardize the features
        std_scaler = preprocessing.StandardScaler()
        X = std_scaler.fit_transform(X)

        # Split the data into train and test sets with an 80-20% ratio
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=random_seed)

    return x_train, x_test, y_train, y_test

