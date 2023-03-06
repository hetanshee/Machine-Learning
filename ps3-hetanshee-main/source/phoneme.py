"""
Author: Hetanshee Shah
Date: 10-15-2022
Description: Perceptron vs Logistic Regression on a Phoneme Dataset
"""

# utilities
from util import *

# scipy libraries
from scipy import stats

# scikit-learn libraries
from sklearn import preprocessing
from sklearn import metrics
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler


######################################################################
# functions
######################################################################

def cv_performance(clf, train_data, kfs):
    """
    Determine classifier performance across multiple trials using cross-validation

    Parameters
    --------------------
        clf        -- classifier
        train_data -- Data, training data
        kfs        -- array of size n_trials
                      each element is one fold from model_selection.KFold

    Returns
    --------------------
        scores     -- numpy array of shape (n_trials, n_fold)
                      each element is the (accuracy) score of one fold in one trial
    """


    scores = np.zeros((kfs, 2))

    ### ========== TODO: START ========== ###
    # part 2: run multiple trials of cross-validation (CV)
    # for each trial, get perf on 1 trial & update scores

    score_val = []
    for i in range(kfs):
        error = cv_performance_one_trial(clf, train_data, kfs)
        scores[i][0], scores[i][1] = i + 1, error
        score_val.append(error)

    score1 = np.mean(score_val)
    score2 = np.std(score_val)

    ### ========== TODO: END ========== ###

    return score1, score2


def cv_performance_one_trial(clf, train_data, kf):
    """
    Compute classifier performance across multiple folds using cross-validation

    Parameters
    --------------------
        clf        -- classifier
        train_data -- Data, training data
        kf         -- model_selection.KFold

    Returns
    --------------------
        scores     -- numpy array of shape (n_fold, )
                      each element is the (accuracy) score of one fold
    """

    ### ========== TODO: START ========== ###
    # part 2: run one trial of cross-validation (CV)
    # for each fold, train on its data, predict, and update score
    # hint: check KFold.split and metrics.accuracy_score

    train_data = load_data('phoneme_train.csv')
    X = train_data.X
    y = train_data.y
    test_error = 0
    count = 0

    kf = KFold(n_splits=10, random_state=np.random.randint(1234), shuffle=True)
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        count = count + 1
        training = clf.fit(X_train, y_train)
        pred = training.predict(X_test)
        test_error += (metrics.accuracy_score(y_test, pred))

    ### ========== TODO: END ========== ###
    # print(count)
    return test_error / count


######################################################################
# main
######################################################################

def main():
    np.random.seed(1234)

    # ========================================
    # load data
    train_data = load_data('phoneme_train.csv')
    X = train_data.X
    y = train_data.y

    ### ========== TODO: START ========== ###
    # part 1: is data linearly separable? Try Perceptron and LogisticRegression
    # hint: set fit_intercept = True to have offset (i.e., bias)
    # hint: you may also want to try LogisticRegression with C=1e10

    x_trial, x_test, y_trial, y_test = train_test_split(X, y, test_size=0.2)
    clf1 = Perceptron(fit_intercept=True)
    clf2 = LogisticRegression(C=1e10, solver='liblinear')
    clf1.fit(x_trial, y_trial)
    clf2.fit(x_trial, y_trial)
    predictions1 = clf1.predict(x_test)
    ac = accuracy_score(y_test, predictions1)
    print("Perceptron accuracy score")
    print(f'{ac:9.3f}')
    predictions2 = clf2.predict(x_test)
    ac = accuracy_score(y_test, predictions2)
    print("Logistic Regression accuracy score")
    print(f'{ac:9.3f}')

    ### ========== TODO: END ========== ###

    ### ========== TODO: START ========== ###
    # parts 3-4: compare classifiers
    # make sure to use same folds across all runs (hint: model_selection.KFold)
    # hint: for standardization, use preprocessing.StandardScaler()

    kfs = 10

    print("classifier  |   µ    |   σ ")
    print("without preprocessing")
    P = Perceptron(fit_intercept=True)
    L = LogisticRegression(fit_intercept=True, C=1e10, solver='liblinear')
    R = LogisticRegression(fit_intercept=True, penalty='l2', C=1, max_iter=1000, solver=
    'liblinear')

    a1, b1 = cv_performance(P, train_data, kfs)
    print(f'    P    {a1:9.3f} {b1:9.3f} ')
    a2, b2 = cv_performance(L, train_data, kfs)
    print(f'    L    {a2:9.3f} {b2:9.3f} ')
    a3, b3 = cv_performance(R, train_data, kfs)
    print(f'    R    {a3:9.3f} {b3:9.3f} ')
    print("With Data standardization")

    scaler = preprocessing.StandardScaler().fit(train_data.X)
    x_train_scaled = scaler.transform(train_data.X)
    train_data = Data(X=x_train_scaled, y=train_data.y)

    P = Perceptron(fit_intercept=True)
    L = LogisticRegression(fit_intercept=True, C=1e10, solver='liblinear')
    R = LogisticRegression(fit_intercept=True, penalty='l2', C=1, solver='liblinear')

    a1, b1 = cv_performance(P, train_data, kfs)
    print(f'    P    {a1:9.3f} {b1:9.3f} ')
    a2, b2 = cv_performance(L, train_data, kfs)
    print(f'    L    {a2:9.3f} {b2:9.3f} ')
    a3, b3 = cv_performance(R, train_data, kfs)
    print(f'    R    {a3:9.3f} {b3:9.3f} ')

    ### ========== TODO: END ========== ###


if __name__ == '__main__':
    main()