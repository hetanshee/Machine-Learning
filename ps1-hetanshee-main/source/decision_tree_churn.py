from sklearn.datasets import load_iris, load_wine
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn import tree
import matplotlib.pyplot as plt
import time
import numpy as np
from scipy import stats
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import os
import csv


class Data:

    def __init__(self):
        """
        Data class.

        Attributes
        --------------------
            X -- numpy array of shape (n,d), features
            y -- numpy array of shape (n,), targets
        """

        # n = number of examples, d = dimensionality
        self.X = None
        self.y = None

        self.Xnames = None
        self.yname = None

    def load(self, filename, header=0, predict_col=-1):
        """Load csv file into X array of features and y array of labels."""

        # determine filename
        dir = os.path.dirname(__file__)
        f = os.path.join(dir, '..', 'data', filename)

        # load data
        with open(f, 'r') as fid:
            data = np.genfromtxt(fid, delimiter=",")

        # separate features and labels
        if predict_col is None:
            self.X = data[:, :]
            self.y = None
        else:
            if data.ndim > 1:
                self.X = np.delete(data, predict_col, axis=1)
                self.y = data[:, predict_col]
            else:
                self.X = None
                self.y = data[:]

        # load feature and label names
        if header != 0:
            with open(f, 'r') as fid:
                header = fid.readline().rstrip().split(",")

            if predict_col is None:
                self.Xnames = header[:]
                self.yname = None
            else:
                if len(header) > 1:
                    self.Xnames = np.delete(header, predict_col)
                    self.yname = header[predict_col]
                else:
                    self.Xnames = None
                    self.yname = header[0]
        else:
            self.Xnames = None
            self.yname = None


# helper functions
def load_data(filename, header=0, predict_col=-1):
    """Load csv file into Data class."""
    data = Data()
    data.load(filename, header=header, predict_col=predict_col)
    return data

def replace_null_values(X):
    nan_indices = np.argwhere(np.isnan(X))
    column_means = np.nanmean(X,axis=0)

    for nan in nan_indices:
        X[nan[0],nan[1]] = column_means[nan[1]]
    return X

def replace_outliers(X, m=3):

    ## Replacing the outlier with the Median of the column.

    for i in range(0,X.shape[1]):
        column_median = np.median(X[:,i])
        # print(o_median)
        outlier_scores = np.abs(stats.zscore(X[:,i]))
        outlier_indices = np.argwhere(outlier_scores>m)

        for outlier in outlier_indices:
            X[outlier,i] = column_median
        # print(X[358,4])

    return X

def main():
    ## Data Collection

    data = load_data("/Users/91799/Downloads/Customer_churn_raw.csv", header=1)

    X = data.X[1:, :]
    y = data.y[1:]
    print('y: ',y)

    print(X.shape)
    print(y.shape)

    replace_outliers(X, m=3)
    replace_null_values(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    # X_train, X_test = np.array(X_train), np.array(X_test)
    # y_train, y_test = np.array(y_train), np.array(y_test)
    # print(y[0])

    clf_dtc = DTC(criterion='entropy', max_depth=5, random_state=1234)
    clf_dtc.fit(X_train, y_train)
    preds = clf_dtc.predict(X_test)
    print("My DT:")
    #clf_dtc.print_tree()
    print("------------------------------------------------------")
    print("My predictions:\n", preds)
    print("------------------------------------------------------")
    # Get the accuracy
    correct_count = len(set(preds) & set(y_test))
    accuracy = correct_count / len(y_test)
    print("Accuracy:\n", accuracy)
    print("------------------------------------------------------")
    # Test w/ customer dataset using SKLearn
    skl_clf = DTC(splitter="best", random_state=1234, max_depth=5)
    skl_clf.fit(X_train, y_train)
    skl_preds = skl_clf.predict(X_test)
    print("SKLearn predictions:\n", skl_preds)
    print("------------------------------------------------------")
    # Get the accuracy
    skl_correct_count = len(set(skl_preds) & set(y_test))
    skl_accuracy = skl_correct_count / len(y_test)
    print("SKL accuracy:\n", skl_accuracy)
    print("------------------------------------------------------")
    # Save an image of the SKL tree
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(4, 4), dpi=300)
    tree.plot_tree(skl_clf, class_names=True);
    fig.savefig('SKL_DT.png')

if __name__ == '__main__':
    main()
