from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
import pandas as pd
import numpy as np

def pick_best_features(X,y,n):
    '''
    Takes in the feature matrix X and y. Performs feature ranking with recursive
    feature elimination using LogisticRegression on first 1 feature, than 2,
    than so on until we use all the features. For instance, when it runs on 3
    features it figures out which 3 features give us the best results and then
    uses those.

    The output is a dictionary with keys of all possible features and the values
    are how many times the feature was used. So the higher the number the better
    signal it has. For instance, 'concavity_worst' was used 31 times, so it was
    used in every model. On the other hand, 'compactness_se' was only used 1
    time so it has the worst signal.
    '''
    rankings = []
    for i in xrange(1,n+1):
        model = LogisticRegression()
        rfe = RFE(model, i)
        rfe.fit(X,y)
        rankings.append(rfe.ranking_)

    rankings = np.array(rankings).T
    features_list = X.columns
    feature_counts = {}

    for i, feature in enumerate(rankings):
        count = 0
        for run in feature:
            if run == 1:
                count += 1
        feature_counts[features_list[i]] = count

    return feature_counts


# see how many features I should use, check my functions file for more indepth explanation
def how_many_features_do_we_want(features, X, y):
    '''
    Takes in a dictionary with the features as keys and how many times they were
    used to model in the above function as values. Runs LogisticRegression models
    starting with one feature in the X, this feature is the most used feature
    from the above function. Then runs the next model with two features, the most
    and second used features, and so on until ever feature is used.

    A dictionary is retured with the keys being how many features are in the
    model and the value is a list consisting of true positive, true negative,
    false positive, and false negative, in that order. The features list in
    order of most to least predictive is also returned.
    '''
    sorted_features = sorted(features, key=features.get, reverse=True)
    a, feature_list = [], []
    for feature in sorted_features:
        a.append(feature)
        b = a[:]
        feature_list.append(b)

    outcome = {}
    for i, feature in enumerate(feature_list):
        small_X = X[feature]
        model = LogisticRegression().fit(small_X,y)
        predict = model.predict(small_X)

        tp, tn, fp, fn = 0, 0, 0, 0
        for num in zip(np.array(y),predict):
            if num == (1,1):
                tp += 1
            elif num == (1,0):
                fn += 1
            elif num == (0,1):
                fp += 1
            elif num == (0,0):
                tn += 1

        outcome[i+1] = [tp,tn,fp,fn]

    return outcome, feature
