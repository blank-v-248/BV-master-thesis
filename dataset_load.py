import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons
import pandas as pd

def synth_data(dimensions=2, num_points=100, random_seed=24):
    """
    Generates a synthetic dataset that has two gaussian clusters & splits it into training and testing sets.

    Args:
        dimensions (int): Number of features/dimensions (must be 2 for the moons dataset).
        random_seed (int): Random seed for reproducibility.
        num_points (int): Number of points per class.

    Returns:
        x_train (ndarray): Training features.
        x_test (ndarray): Testing features.
        y_train (ndarray): Training labels.
        y_test (ndarray): Testing labels.
    """
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

def synth_data_moons(dimensions, random_seed, num_points):
    """
    Generates a synthetic moons dataset, splits it into training and testing sets.

    Args:
        dimensions (int): Number of features/dimensions (must be 2 for the moons dataset).
        random_seed (int): Random seed for reproducibility.
        num_points (int): Number of points per class.

    Returns:
        x_train (ndarray): Training features.
        x_test (ndarray): Testing features.
        y_train (ndarray): Training labels.
        y_test (ndarray): Testing labels.
    """
    if dimensions != 2:
        raise ValueError("The moons dataset only supports 2 dimensions.")

    # Generate the dataset
    X, y = make_moons(n_samples=num_points, noise=0.3, random_state=random_seed)

    # Split into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=random_seed
    )

    return x_train, x_test, y_train, y_test

def loan_data(train_val: bool = False):
    """
    Loads the loan dataset and splits it into training and testing sets.

    Args:
        train_val (bool): Whether the training set should include the validation set, default: False

    Returns:
        x_train (ndarray): Training features.
        x_test (ndarray): Testing features.
        y_train (ndarray): Training labels.
        y_test (ndarray): Testing labels.
    """
    six_most_significant_features = ['AvailableBankcardCredit', 'LoanOriginalAmount',
                                     'TradesNeverDelinquent(percentage)',
                                     'BankcardUtilization', 'TotalInquiries', 'CreditHistoryLength']
    if train_val:
        x_train0 = pd.read_csv('data/train_val_pre2009_f_star_loan_status.csv')
    else:
        x_train0=pd.read_csv('data/train_pre2009_f_star_loan_status.csv')
    x_train=x_train0[six_most_significant_features].to_numpy()
    y_train=x_train0["LoanStatus"].to_numpy()
    y_train[y_train == -1] = 0

    x_test0=pd.read_csv('data/test_pre2009_f_star_loan_status.csv')
    x_test=x_test0[six_most_significant_features].to_numpy()
    y_test=x_test0["LoanStatus"].to_numpy()
    y_test[y_test == -1] = 0

    return x_train, x_test, y_train, y_test




