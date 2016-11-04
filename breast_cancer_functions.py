from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
import pandas as pd
import numpy as np
from sklearn.grid_search import GridSearchCV

class Grid_Search_All:
    '''
    Class to run GridSearchCV simultaneously on different models
    and any hyperparameters you choose for them. Then displays
    the outcome.

    has two methods:

    fit - runs all of the GridSearchCV
    score_summary - displays scores in an ordered pandas dataframe
    '''
    def __init__(self, models, params):
        if not set(models.keys()).issubset(set(params.keys())):
            missing_params = list(set(models.keys()) - set(params.keys()))
            raise ValueError("Some estimators are missing parameters: %s" % missing_params)

        self.models = models
        self.params = params
        self.keys = models.keys()
        self.gridsearches = {}

    def fit(self, X, y, cv=5, pre_dispatch=4, refit=False):
        '''
        Fits all of the models with all of the parameter options
        using cross validation.

        cv = crossvalidation, default is 5
        pre_dispatch = number of jobs run in parallel, default is 4 because
                       my computer has 4 cores
        refit = whether or not it will fit all data to best model from
                crossvalidation, default is False because I don't need
                it so it would waste time
        '''
        for model_name in self.keys:
            print "Running GridSearchCV for {}'s.".format(model_name)
            model = self.models[model_name]
            par = self.params[model_name]

            grid_search = GridSearchCV(model, par, cv=cv, pre_dispatch=pre_dispatch, refit=refit)
            grid_search.fit(X,y)

            self.gridsearches[model_name] = grid_search

    def score_summary(self, sort_by='mean_score'):
        '''
        This builds and prints a pandas dataframe of the summary of all the
        different fits of the models and orders them by best performing
        in a category that you tell it to.
        '''
        def row(key, scores, params):
            d = {'estimator': key,
                 'min_score': np.min(scores),
                 'max_score': np.max(scores),
                 'mean_score': np.mean(scores),
                 'std_score': np.std(scores)
                }
            return pd.Series(dict(params.items() + d.items()))

        rows = []
        for k in self.keys:
            for gsc in self.gridsearches[k].grid_scores_:
                rows.append(row(k, gsc.cv_validation_scores, gsc.parameters))

        df = pd.concat(rows, axis=1).T.sort([sort_by], ascending=False)
        columns = ['estimator', 'min_score', 'mean_score', 'max_score', 'std_score']
        columns = columns + [c for c in df.columns if c not in columns]
        return df[columns]


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
