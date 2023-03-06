"""
Author: Hetanshee Shah
Date: 09/19/2022
Description: Decision Tree Classifier
Collaborator: Jeet Desai
"""

# Use only the provided packages!
import math
import csv

import matplotlib.pyplot as plt
from util import *

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from collections import Counter
from sklearn.tree import plot_tree

######################################################################
# classes
######################################################################

class Classifier(object):
    """
    Classifier interface.
    """

    def fit(self, X, y):
        raise NotImplementedError()

    def predict(self, X):
        raise NotImplementedError()


class MajorityVoteClassifier(Classifier):

    def __init__(self):
        """
        A classifier that always predicts the majority class.

        Attributes
        --------------------
            prediction_ -- majority class
        """
        self.prediction_ = None

    def fit(self, X, y):
        """
        Build a majority vote classifier from the training set (X, y).

        Parameters
        --------------------
            X    -- numpy array of shape (n,d), samples
            y    -- numpy array of shape (n,), target classes

        Returns
        --------------------
            self -- an instance of self
        """
        vals, counts = np.unique(y, return_counts=True)
        majority_val, majority_count = max(zip(vals, counts), key=lambda val_count: val_count[1])
        self.prediction_ = majority_val
        return self

    def predict(self, X):
        """
        Predict class values.

        Parameters
        --------------------
            X    -- numpy array of shape (n,d), samples

        Returns
        --------------------
            y    -- numpy array of shape (n,), predicted classes
        """
        if self.prediction_ is None:
            raise Exception('Classifier not initialized. Perform a fit first.')

        n,d = X.shape
        y = [self.prediction_] * n
        return y


class RandomClassifier(Classifier):

    def __init__(self):
        """
        A classifier that predicts according to the distribution of the classes.

        Attributes
        --------------------
            probabilities_ -- class distribution dict (key = class, val = probability of class)
        """
        self.probabilities_ = None

    def fit(self, X, y):
        """
        Build a random classifier from the training set (X, y).

        Parameters
        --------------------
            X    -- numpy array of shape (n,d), samples
            y    -- numpy array of shape (n,), target classes

        Returns
        --------------------
            self -- an instance of self
        """

        ### ========== TODO: START ========== ###
        # insert your RandomClassifier code
        cur_dist = Counter(y)
        self.probabilities_ = (float(cur_dist[0.0])/(float(cur_dist[0.0])+float(cur_dist[1.0])))

        ### ========== TODO: END ========== ###

        return self

    def predict(self, X, seed=1234):
        """
        Predict class values.
        Parameters
        --------------------
            X    -- numpy array of shape (n,d), samples
            seed -- integer, random seed

        Returns
        --------------------
            y    -- numpy array of shape (n,), predicted classes
        """
        if self.probabilities_ is None:
            raise Exception('Classifier not initialized. Perform a fit first.')
        np.random.seed(seed)

        ### ========== TODO: START ========== ###
        # insert your RandomClassifier code

        y = np.random.choice(2, X.shape[0], p=[self.probabilities_, 1-self.probabilities_])

        ### ========== TODO: END ========== ###

        return y


######################################################################
# functions
######################################################################

def error(clf, X, y, ntrials=100, test_size=0.2):
    """
    Computes the classifier error over a random split of the data,
    averaged over ntrials runs.

    Parameters
    --------------------
        clf         -- classifier
        X           -- numpy array of shape (n,d), features values
        y           -- numpy array of shape (n,), target classes
        ntrials     -- integer, number of trials
        test_size   -- float (between 0.0 and 1.0) or int,
                       if float, the proportion of the dataset to include in the test split
                       if int, the absolute number of test samples

    Returns
    --------------------
        train_error -- float, training error
        test_error  -- float, test error
    """

    ### ========== TODO: START ========== ###
    # part b: compute cross-validation error over ntrials
    # hint: use train_test_split (be careful of the parameters)

    train_error = 0
    test_error = 0
    
    for i in range(0, ntrials):
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=ntrials)
    
        clf.fit(X_train, y_train)
    
        y_train_predict = clf.predict(X_train)
    
        y_test_predict = clf.predict(X_test)
    
        train_error += 1 - metrics.accuracy_score(y_train, y_train_predict, normalize=True)
    
        test_error += 1 - metrics.accuracy_score(y_test, y_test_predict, normalize=True)
    
    train_error = train_error/ntrials
    test_error = test_error/ntrials
    
    

    ### ========== TODO: END ========== ###

    return train_error, test_error


def write_predictions(y_pred, filename, yname=None):
    """Write out predictions to csv file."""
    out = open(filename, 'w')
    f = csv.writer(out)
    if yname:
        f.writerow([yname])
    f.writerows(zip(y_pred))
    out.close()


######################################################################
# main
######################################################################

def main():
    # load Titanic dataset
    titanic = load_data('titanic_train.csv', header=1, predict_col=0)
    X = titanic.X; Xnames = titanic.Xnames
    y = titanic.y; yname = titanic.yname
    n,d = X.shape  # n = number of examples, d =  number of features



    #========================================
    # train Majority Vote classifier on data
    print('Classifying using Majority Vote...')
    maj_clf = MajorityVoteClassifier() # create MajorityVote classifier, which includes all model parameters
    maj_clf.fit(X, y)                  # fit training data using the classifier
    y_pred = maj_clf.predict(X)        # take the classifier and run it on the training data
    train_error = 1 - metrics.accuracy_score(y, y_pred, normalize=True)
    print('\t-- training error: %.3f' % train_error)
    
    print('Classifying using Random Classifier...')
    rand_clf = RandomClassifier()
    rand_clf.fit(X, y)
    y_pred = rand_clf.predict(X)
    train_error = 1 - metrics.accuracy_score(y, y_pred, normalize=True)
    print('\t -- training error: %.3f' % train_error)



    ### ========== TODO: START ========== ###
    # part a: evaluate training error of Decision Tree classifier
    print('Classifying using Decision Tree...')
    dec_clf = DecisionTreeClassifier(criterion='entropy', random_state=11)
    dec_clf.fit(X, y)
    y_pred = dec_clf.predict(X)
    train_error = 1 - metrics.accuracy_score(y, y_pred, normalize=True)
    print('\t-- training error: %.3f' % train_error)

    ### ========== TODO: END ========== ###



    # note: uncomment out the following lines to output the Decision Tree graph
    
    # save the classifier -- requires GraphViz and pydot
       # import pydotplus, pydot
       # import matplotlib.pyplot as plt
       # from io import StringIO
       # import io
       # from sklearn import tree

        #dot_data = io.StringIO()
        #tree.export_graphviz(dec_clf, out_file=dot_data, feature_names=Xnames, class_names=["Died", "Survived"])
        #pydotplus.graph_from_dot_data(dot_data.getvalue())
    
    
    #dec_tree = plot_tree(decision_tree=dec_clf, feature_names = Xnames,
    #          class_names =["Died", "Survived"] , filled = True , precision = 4, rounded = True)
    # plt.plot(block=True)

    ### ========== TODO: START ========== ###
    # part b: use cross-validation to compute average training and test error of classifiers
    
    print('Investigating various classifiers...')
    
    avg_maj_train_error, avg_maj_test_error = error(maj_clf, X, y)
    print('For Majority Vote classifier')
    print('\t -- Training Error : %.3f' % avg_maj_train_error)
    print('\t -- Testing Error : %.3f' % avg_maj_test_error)
    
    avg_ran_train_error, avg_ran_test_error = error(rand_clf, X, y)
    print('For Random Classifier')
    print('\t -- Training Error : %.3f' % avg_ran_train_error)
    print('\t -- Testing Error : %.3f' % avg_ran_test_error)
    
    avg_dec_train_error, avg_dec_test_error = error(dec_clf, X, y)
    print('For Decision Tree classifier')
    print('\t -- Training Error : %.3f' % avg_dec_train_error)
    print('\t -- Testing Error : %.3f' % avg_dec_test_error)

    ### ========== TODO: END ========== ###



    ### ========== TODO: START ========== ###
    # part c: investigate decision tree classifier with various depths
    print('Investigating depths...')
    dec_train_error=[]
    dec_test_error=[]
    for i in range(1, 21):
        cur_dec_clf = DecisionTreeClassifier(criterion="entropy", max_depth=i)
        dec_train, dec_test = error(cur_dec_clf, X, y)
        dec_train_error.append(dec_train)
        dec_test_error.append(dec_test)
        print('\t-- error for %g-depth decision tree... --training error: %.3f & --testing error: %.3f' % (i, dec_train, dec_test))

    plt.plot(range(1,21), dec_test_error, label='Decision Test error', color='r')
    plt.plot(range(1, 21), dec_train_error, label='Decision Train error', color='r', linestyle='dashed')
    plt.plot(range(1,21), [avg_maj_test_error]*20, label='Majority Test Error', color='b')
    plt.plot(range(1,21), [avg_ran_test_error]*20, label='Random Test Error', color='g')

    plt.xlabel('Depth')
    plt.ylabel('Error')
    plt.legend()
    plt.axis([1, 20, 0, 1])
    plt.show(block=True)
    ### ========== TODO: END ========== ###



    ### ========== TODO: START ========== ###
    # part d: investigate decision tree classifier with various training set sizes
    print('Investigating training set sizes...')
    
    dec_tree_clf = DecisionTreeClassifier(criterion = 'entropy', max_depth=9)
    maj_vot_clf = MajorityVoteClassifier()
    ran_clf = RandomClassifier()
    split_list = np.arange(0.05, 1, 0.05)
    print(split_list)
    train_error_list = []
    test_error_list = []
    maj_test_error = []
    ran_test_error = []
    for x in split_list:
        train_error, test_error = error(dec_tree_clf, X, y, test_size=x)
        train_error_list.append(train_error)
        test_error_list.append(test_error)
        m_train_error, m_test_error = error(maj_vot_clf, X,y,test_size=x)
        maj_test_error.append(m_test_error)
        r_train_error, r_test_error = error(ran_clf, X, y, test_size=x)
        ran_test_error.append(r_test_error)

    print(train_error_list)
    print(test_error_list)

    plt.plot(split_list, train_error_list, label = 'Decision Training error', color='r', linestyle='dashed')
    plt.plot(split_list, test_error_list, label = 'Decision Testing error',  color='r')
    plt.plot(split_list, [np.sum(maj_test_error)/19]*19, label = 'Majority Testing Error',  color='b')
    plt.plot(split_list, [np.sum(ran_test_error)/19]*19, label = 'Random Testing Error',  color='g')
    plt.xlabel('Test Size')
    plt.ylabel('Error')
    plt.legend()
    plt.xlim((0.05, 0.95))
    plt.ylim((0, 1))
    plt.show(block = True)

    ### ========== TODO: END ========== ###


    ### ========== TODO: START ========== ###
    # Contest
    # uncomment write_predictions and change the filename

    # evaluate on test data

    for i in range(1, 10):
        cur_dec_clf = DecisionTreeClassifier(criterion = "entropy", max_depth = i, random_state= 11, max_features = 'sqrt' )
        clf_err = error(cur_dec_clf, X, y)
        print('\t-- error for %g-depth decision tree... --training error: %.3f & --testing error: %.3f' % (i, clf_err[0], clf_err[1]))


    titanic_test = load_data('titanic_test.csv', header=1, predict_col=None)
    X_test = titanic_test.X
    y_pred = cur_dec_clf.predict(X_test)   # take the trained classifier and run it on the test data
    write_predictions(y_pred, 'C:/Users/91799/Downloads/ps1-hetanshee-main/ps1-hetanshee-main/data/hetanshee_titanic.csv', titanic.yname)

    ### ========== TODO: END ========== ###



    print('Done')


if __name__ == '__main__':
    main()
