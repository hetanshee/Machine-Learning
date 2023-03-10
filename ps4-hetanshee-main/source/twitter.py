"""
Author: JEET
Date: 10/29/2022
Description: Twitter
"""

import numpy as np

from string import punctuation

from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
from sklearn.utils import shuffle
import matplotlib.pyplot as plt


######################################################################
# functions -- input/output
######################################################################

def read_vector_file(fname):
    """
    Reads and returns a vector from a file.

    Parameters
    --------------------
        fname  -- string, filename

    Returns
    --------------------
        labels -- numpy array of shape (n,)
                    n is the number of non-blank lines in the text file
    """
    return np.genfromtxt(fname)


def write_label_answer(vec, outfile):
    """
    Writes your label vector to the given file.

    Parameters
    --------------------
        vec     -- numpy array of shape (n,) or (n,1), predicted scores
        outfile -- string, output filename
    """

    # for this project, you should predict 70 labels
    if(vec.shape[0] != 70):
        print("Error - output vector should have 70 rows.")
        print("Aborting write.")
        return

    np.savetxt(outfile, vec)


######################################################################
# functions -- feature extraction
######################################################################

def extract_words(input_string):
    """
    Processes the input_string, separating it into "words" based on the presence
    of spaces, and separating punctuation marks into their own words.

    Parameters
    --------------------
        input_string -- string of characters

    Returns
    --------------------
        words        -- list of lowercase "words"
    """

    for c in punctuation:
        input_string = input_string.replace(c, ' ' + c + ' ')
    return input_string.lower().split()


def extract_dictionary(infile):
    """
    Given a filename, reads the text file and builds a dictionary of unique
    words/punctuations.

    Parameters
    --------------------
        infile    -- string, filename

    Returns
    --------------------
        word_list -- dictionary, (key, value) pairs are (word, index)
    """

    word_list = {}
    index = 0 
    with open(infile, 'r') as fid:
        
        ### ========== TODO: START ========== ###
        # part 1-1: process each line to populate word_list
        
         for lines in fid:
            extract = extract_words(lines)
            for words in extract:
                if words not in word_list.keys(): 
                    word_list[words] = index
                    index += 1
                
        #print(word_list)
    with open('dict.tsv', 'w') as f:
            for key, value in word_list.items(): 
                f.write('%s\t%s\n' % (value, key))
         
    
        
        ### ========== TODO: END ========== ###

    return word_list


def extract_feature_vectors(infile, word_list):
    """
    Produces a bag-of-words representation of a text file specified by the
    filename infile based on the dictionary word_list.

    Parameters
    --------------------
        infile         -- string, filename
        word_list      -- dictionary, (key, value) pairs are (word, index)

    Returns
    --------------------
        feature_matrix -- numpy array of shape (n,d)
                          boolean (0,1) array indicating word presence in a string
                            n is the number of non-blank lines in the text file
                            d is the number of unique words in the text file
    """

    num_lines = sum(1 for line in open(infile,'r'))
    num_words = len(word_list)
    feature_matrix = np.zeros((num_lines, num_words))

    with open(infile, 'r') as fid:
        ### ========== TODO: START ========== ###
        # part 1-2: process each line to populate feature_matrix
        
        for index,lines in enumerate(fid):
            extract = extract_words(lines)
            for i in extract:
                if i not in word_list:
                    continue
                d_index = word_list[i]
                
                feature_matrix[index,d_index] = 1
       
        ### ========== TODO: END ========== ###

    return feature_matrix


def test_extract_dictionary(dictionary):
    err = 'extract_dictionary implementation incorrect'

    assert len(dictionary) == 1811

    exp = [('2012', 0),
           ('carol', 10),
           ('ve', 20),
           ('scary', 30),
           ('vacation', 40),
           ('just', 50),
           ('excited', 60),
           ('no', 70),
           ('cinema', 80),
           ('frm', 90)]
    act = [sorted(dictionary.items(), key=lambda it: it[1])[i] for i in range(0, 100, 10)]
    assert exp == act


def test_extract_feature_vectors(X):
    err = 'extract_features_vectors implementation incorrect'

    assert X.shape == (630, 1811)

    exp = np.array([[ 1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.],
                    [ 1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.],
                    [ 0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  1.],
                    [ 0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,  1.,  1.],
                    [ 0.,  1.,  0.,  0.,  0.,  1.,  0.,  0.,  1.,  1.],
                    [ 0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  1.,  1.],
                    [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  1.],
                    [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  1.],
                    [ 0.,  1.,  0.,  0.,  1.,  0.,  0.,  0.,  1.,  1.],
                    [ 0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.]])
    act = X[:10,:10]
    assert (exp == act).all()


######################################################################
# functions -- evaluation
######################################################################

def performance(y_true, y_pred, metric='accuracy'):
    """
    Calculates the performance metric based on the agreement between the
    true labels and the predicted labels.

    Parameters
    --------------------
        y_true -- numpy array of shape (n,), known labels
        y_pred -- numpy array of shape (n,), (continuous-valued) predictions
        metric -- string, option used to select the performance measure
                  options: 'accuracy', 'f1_score', 'auroc', 'precision',
                           'sensitivity', 'specificity'

    Returns
    --------------------
        score  -- float, performance score
    """
    # map continuous-valued predictions to binary labels
    y_label = np.sign(y_pred)
    y_label[y_label==0] = 1 # map points of hyperplane to +1

    ### ========== TODO: START ========== ###
    # part 2-1: compute classifier performance with sklearn metrics
    # hint: sensitivity == recall
    # hint: use confusion matrix for specificity (use the labels param)
   
    if metric =='accuracy':
        score = metrics.accuracy_score(y_true,y_label)
    elif metric == 'f1_score':
        score = metrics.f1_score(y_true, y_label)
    elif metric == 'auroc':
        score = metrics.roc_auc_score(y_true, y_label)
    elif metric == 'precision':
        score = metrics.precision_score(y_true, y_label)
    elif metric == 'sensitivity':
        conf_mat = metrics.confusion_matrix(y_true, y_label)
        score = conf_mat[1,1]/float((conf_mat[1,1]+conf_mat[1, 0]))
    elif metric == 'specificity':
        conf_mat = metrics.confusion_matrix(y_true, y_label)
        score = conf_mat[0,0]/float((conf_mat[0,0]+conf_mat[0,1]))
    else:
        print('wrong metrics')
    
   
    return score
    ### ========== TODO: END ========== ###


def test_performance():
    """Ensures performance scores are within epsilon of correct scores."""

    y_true = [ 1,  1, -1,  1, -1, -1, -1,  1,  1,  1]
    y_pred = [ 3.21288618, -1.72798696,  3.36205116, -5.40113156,  6.15356672,
               2.73636929, -6.55612296, -4.79228264,  8.30639981, -0.74368981]
    metrics = ['accuracy', 'f1_score', 'auroc', 'precision', 'sensitivity', 'specificity']
    scores  = [     3/10.,      4/11.,   5/12.,        2/5.,          2/6.,          1/4.]

    import sys
    eps = sys.float_info.epsilon

    for i, metric in enumerate(metrics):
        assert abs(performance(y_true, y_pred, metric) - scores[i]) < eps, \
            (metric, performance(y_true, y_pred, metric), scores[i])


def cv_performance(clf, X, y, kf, metric='accuracy'):
    """
    Splits the data, X and y, into k folds and runs k-fold cross-validation.
    Trains classifier on k-1 folds and tests on the remaining fold.
    Calculates the k-fold cross-validation performance metric for classifier
    by averaging the performance across folds.

    Parameters
    --------------------
        clf    -- classifier (instance of SVC)
        X      -- numpy array of shape (n,d), feature vectors
                    n = number of examples
                    d = number of features
        y      -- numpy array of shape (n,), binary labels {1,-1}
        kf     -- model_selection.KFold or model_selection.StratifiedKFold
        metric -- string, option used to select performance measure

    Returns
    --------------------
        score   -- float, average cross-validation performance across k folds
    """

    scores = []
    for train, test in kf.split(X, y):
        X_train, X_test, y_train, y_test = X[train], X[test], y[train], y[test]
        clf.fit(X_train, y_train)
        # use SVC.decision_function to make "continuous-valued" predictions
        y_pred = clf.decision_function(X_test)
        score = performance(y_test, y_pred, metric)
        if not np.isnan(score):
            scores.append(score)
    return np.array(scores).mean()


def select_param_linear(X, y, kf, metric='accuracy'):
    """
    Sweeps different settings for the hyperparameter of a linear-kernel SVM,
    calculating the k-fold CV performance for each setting, then selecting the
    hyperparameter that maximizes the average k-fold CV performance.

    Parameters
    --------------------
        X      -- numpy array of shape (n,d), feature vectors
                    n = number of examples
                    d = number of features
        y      -- numpy array of shape (n,), binary labels {1,-1}
        kf     -- model_selection.KFold or model_selection.StratifiedKFold
        metric -- string, option used to select performance measure

    Returns
    --------------------
        C -- float, optimal parameter value for linear-kernel SVM
    """

    print('Linear SVM Hyperparameter Selection based on ' + str(metric) + ':')
    C_range = 10.0 ** np.arange(-3, 3)

    ### ========== TODO: START ========== ###
    # part 2-3: select optimal hyperparameter using cross-validation
    # hint: create a new sklearn linear SVC for each value of C
   
    score = []
    for i in C_range:
        #print(i)
        clf = SVC(kernel="linear", C=i)
        scores = cv_performance(clf, X, y, kf, metric=metric)
        #print(i, scores)
        score.append(scores)
        
    print(str(metric) + ':', score)
    max_index = score.index(max(score))  
    
    return C_range[max_index]
   
    
   
    ### ========== TODO: END ========== ###


def select_param_rbf(X, y, kf, metric='accuracy'):
    """
    Sweeps different settings for the hyperparameters of an RBF-kernel SVM,
    calculating the k-fold CV performance for each setting, then selecting the
    hyperparameters that 'maximize' the average k-fold CV performance.

    Parameters
    --------------------
        X       -- numpy array of shape (n,d), feature vectors
                     n = number of examples
                     d = number of features
        y       -- numpy array of shape (n,), binary labels {1,-1}
        kf      -- model_selection.KFold or model_selection.StratifiedKFold
        metric  -- string, option used to select performance measure

    Returns
    --------------------
        gamma, C -- tuple of floats, optimal parameter values for an RBF-kernel SVM
    """

    print('RBF SVM Hyperparameter Selection based on ' + str(metric) + ':')

    ### ========== TODO: START ========== ###
    # (Optional) part 3-1: create grid, then select optimal hyperparameters using cross-validation
    C_range = 10.0 ** np.arange(-3, 3)
    gamma_range = 10.0 ** np.arange(-3, 3)
    max_score = 0
    best_C = 0
    best_gamma = 0
    for i in C_range:
        for gamma in gamma_range:
            score = cv_performance(SVC(C=i,kernel='rbf',gamma=gamma),X=X, y=y,kf=kf,metric=metric)
            if score > max_score:
                max_score = score
                best_C = i
                best_gamma = gamma
    
    print("Maximum score for", metric, " : ", max_score)
    print("----------------------------------------------------------------------------")
    return best_C, best_gamma
   
   
   
    ### ========== TODO: END ========== ###


def performance_CI(clf, X, y, metric='accuracy'):
    """
    Estimates the performance of the classifier using the 95% CI.

    Parameters
    --------------------
        clf          -- classifier (instance of SVC)
                          [already fit to data]
        X            -- numpy array of shape (n,d), feature vectors of test set
                          n = number of examples
                          d = number of features
        y            -- numpy array of shape (n,), binary labels {1,-1} of test set
        metric       -- string, option used to select performance measure

    Returns
    --------------------
        score        -- float, classifier performance
        lower, upper -- tuple of floats, confidence interval
    """

    y_pred = clf.decision_function(X)
    score = performance(y, y_pred, metric)

    ### ========== TODO: START ========== ###
    # part 4-2: use bootstrapping to compute 95% confidence interval
    # hint: use np.random.randint(...) to get a random sample from y
    # hint: lower and upper are the values at 2.5% and 97.5% of the scores
   
    n = X.shape[0]
   
    B = 1000
    lower = 25
    upper = 975
    scores = []

    for i in range(B):
        sampel = []
        for j in range(n):
            sampel.append(np.random.randint(0,n-1))
        
        sample_pred = clf.decision_function(X[sampel])
        
        y_sample = y[sampel]
        score = performance(y_sample, sample_pred, metric=metric)
        scores.append(score)

    scores.sort()

    Mean = np.float64(np.mean(scores))
    lower = np.float64(scores[lower])
    upper = np.float64(scores[upper])

    return Mean, lower, upper  
   
  
    ### ========== TODO: END ========== ###


######################################################################
# main
######################################################################

def main():
    # read the tweets and its labels
    dictionary = extract_dictionary('../data/tweets.txt')
    test_extract_dictionary(dictionary)
    X = extract_feature_vectors('../data/tweets.txt', dictionary)
    test_extract_feature_vectors(X)
    y = read_vector_file('../data/labels.txt')

    # shuffle data (since file has tweets ordered by movie)
    X, y = shuffle(X, y, random_state=0)

    # set random seed
    np.random.seed(1234)

    # split the data into training (training + cross-validation) and testing set
    X_train, X_test = X[:560], X[560:]
    y_train, y_test = y[:560], y[560:]

    metric_list = ['accuracy', 'f1_score', 'auroc', 'precision', 'sensitivity', 'specificity']

    ### ========== TODO: START ========== ###
    # test_performance()

    # part 2-2: create stratified folds (5-fold CV)

    max_score = {}
    print('Linear SVM Hyperparameter Selection')
    print("--------------------------------------------------")
    C_range = 10.0 ** np.arange(-3, 3)
    print("C_values :", C_range)
    print("-----------------------------------------------------")
    skf = StratifiedKFold(n_splits=5,shuffle=True) 
    for i in metric_list:
        C_max = select_param_linear(X_train, y_train, skf,metric=i)
        max_score[i] = C_max
    print("-------------------------------------------------------------")    
    print(max_score)
    print("-------------------------------------------------------------")

    # part 2-4: for each metric, select optimal hyperparameter for linear-kernel SVM using CV

     # (Optional) part 3-2: for each metric, select optimal hyperparameter for RBF-SVM using CV

    C_gamma_dict = {}
    for i in metric_list:
        best_C, best_gamma = select_param_rbf(X_train,y_train,skf,metric=i)
        C_gamma_dict[i] = best_C, best_gamma
        
    print(C_gamma_dict)
    print("--------------------------------------------------------------------")
    print("Best C and gamma",100,0.01)
    ("------------------------------------------------------------------------")
    
    c = 100
    gamma = 0.01

    # part 4-1: train linear-kernal SVM with selected hyperparameters

    clf_linear = SVC(C=1.0,kernel='linear')
    clf_rbf = SVC(C=c,kernel='rbf',gamma=gamma)
    clf_linear.fit(X_train,y_train)
    clf_rbf.fit(X_train,y_train)

    
   
    # part 4-3: use bootstrapping to report performance on test data

    print("For Linear kernal SVM")
    print("---------------------------------------------------------------------------")
    for i in metric_list:
        Mean, lower, upper = performance_CI(clf_linear, X_test, y_test, metric=i)
        print(i," : ",Mean, lower, upper)
    print("---------------------------------------------------------------------------")    
    print("For RBF kernal SVM")
    print("---------------------------------------------------------------------------")
    for i in metric_list:
        Mean, lower, upper = performance_CI(clf_rbf, X_test, y_test, metric=i)
        print(i," : ",Mean, lower, upper)
    print("---------------------------------------------------------------------------")



 
 # part 5: identify important features (hint: use best_clf.coef_[0])



        

    coef = clf_linear.coef_.ravel()
    print('Coefficients: ', clf_linear.coef_.ravel())
    top_positive_coefficients = np.argsort(coef)[-10:]
    print('Top positive coefficients', top_positive_coefficients)
    top_negative_coefficients = np.argsort(coef)[:10]
    print('Top negative coefficients', top_negative_coefficients)
    top_coefficients = np.hstack(
            [top_negative_coefficients, top_positive_coefficients])
    print('Top coefficients\n\n', top_coefficients)

    pos_words = []
    for pos_coef in top_positive_coefficients:
            pos_words.append(list(dictionary.keys())[pos_coef])
            print(list(dictionary.keys())[pos_coef])


    neg_words = []
    for neg_coef in top_negative_coefficients:
            neg_words.append(list(dictionary.keys())[neg_coef])
            print(list(dictionary.keys())[neg_coef])

        # plt.figure(figsize=(15, 5))
    colors = ['red' if c < 0 else 'blue' for c in coef[top_coefficients]]
    plt.bar(np.arange(20), coef[top_coefficients], color=colors)
        # feature_names = np.array(feature_names)
    plt.xticks(np.arange(1, 21),
                neg_words + pos_words, rotation=60, ha='right')
    plt.show(block=True)

        ### ========== TODO: END ========== ###

    ### ========== TODO: START ========== ###
    # part 6: (optional) contest!
    # uncomment out the following, and be sure to change the filename
   
    stop_words = ["a", "an", "and", "are", "as", "at", "be", "but", "by","for", "if", "in", "into", "is", "it","no", "not", "of", "on", "or", "such",
                    "that", "the", "their", "then", "there", "these","they", "this", "to", "was", "will", "with"]
    def NewDict(infile):
            word_list = {}
            index = 0
            with open(infile, 'r') as fid:
               
                for lines in fid:
                    extract = extract_words(lines)
                    for words in extract:
                        if words not in word_list.keys(): 
                            if words not in stop_words:
                                word_list[words] = index
                                index += 1
                return word_list
            
  
  
    def NewV(infile, word_list):
                                
        num_lines = sum(1 for line in open(infile,'r'))
        
        num_words = len(word_list)
        
        feature_matrix = np.zeros((num_lines, num_words))

        with open(infile, 'r') as fid:
       
            for index,lines in enumerate(fid):
                extract = extract_words(lines)
                for i in extract:
                    if i not in word_list:
                        continue
                    d_index = word_list[i]
                    feature_matrix[index,d_index] = 1

            return feature_matrix
        
    
    
                    
    
    svm = SVC(C=100,kernel='rbf',gamma=0.01)
       
    NewD = NewDict('../data/tweets.txt')
    X = NewV('../data/tweets.txt', NewD)
    #     X_train, X_test = X[:560], X[560:]
    #     y_train, y_test = y[:560], y[560:]
    svm.fit(X, y)

       
    X_held = NewV('../data/held_out_tweets.txt',NewD)
    y_pred = svm.decision_function(X_held)
    y_label = np.sign(y_pred)
    y_label[y_label==0] = 1 
       
    write_label_answer(y_label.astype(np.int_), '../data/JEET_twitter.txt')
                    
    """
    X_held = extract_feature_vectors('../data/held_out_tweets.txt', dictionary)
    y_pred = best_clf.decision_function(X_held)
    write_label_answer(y_pred, '../data/YOUR_USERNAME_twitter.txt')
    """
    ### ========== TODO: END ========== ###


if __name__ == '__main__':
    main()
