import pandas as pd
import sqlite3
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.metrics import precision_score, recall_score, f1_score

from breast_cancer_functions import pick_best_features, how_many_features_do_we_want

conn = sqlite3.connect('breast_cancer.db')
c = conn.cursor()

df = pd.read_sql('''SELECT *
                    FROM cancer''', conn)

# this gets run when I'm done working for the session
conn.close()

# this is the X I will use
all_ = list(df.columns[2:])
X = df[all_]
X = X.assign(const=1)

# make the y out of the diagnosis column, this can be used for all of the dataframes
y = [1 if diag == 'M' else 0 for diag in df.diagnosis]

# this scales the data around 0 so no one feature takes over
X[X.columns] = StandardScaler().fit_transform(X)

# check features for how useful they are, check my functions file for more indepth explanation
num_features_to_check = X.shape[1]
features_ranking = pick_best_features(X, y, num_features_to_check)

# see how many features I should use, check my functions file for more indepth explanation
results, features = how_many_features_do_we_want(features_ranking, X, y)

# this is the X I will be working with
X = X[features[:16]]

TP, TN, FP, FN = [],[],[],[]
precision, recall, f1 = [],[],[]
for _ in xrange(100):
    # split into training and testing
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

    # model
    model = LinearSVC(C=0.01).fit(X_train,y_train)
    scores = model.decision_function(X_test)

    # this is what I use to skew results away from false_negative
    # the default in sklearn for this number is 0.0
    threshold = -0.13
    # threshold = 0.0

    prediction = (scores > threshold).astype(np.int)

    tp, tn, fp, fn = 0, 0, 0, 0
    for num in zip(y_test, prediction):
        if num == (1,1):
            tp += 1
        elif num == (1,0):
            fn += 1
        elif num == (0,1):
            fp += 1
        elif num == (0,0):
            tn += 1

    TP.append(tp)
    TN.append(tn)
    FP.append(fp)
    FN.append(fn)
    precision.append(precision_score(y_test, prediction, average='binary'))
    recall.append(recall_score(y_test, prediction, average='binary'))
    f1.append(f1_score(y_test, prediction))

d = {'true_positive':TP,'true_negative':TN,'false_positive':FP,'false_negative':FN,
     'precision':precision,'recall':recall,'f1_score':f1}
outcome_df = pd.DataFrame(data=d)

print '#' * 40
print '#' * 40
print 'Model is: LinearSVC(C=0.01)'
print '   Number of test cases: {}'.format(len(y_test))
print 'Average False Positives: {}'.format(outcome_df.false_positive.mean())
print 'Average False Negatives: {}'.format(outcome_df.false_negative.mean())
print '        Average FN Rate: {}%'.format(round(outcome_df.false_negative.mean()/len(y_test)*100, 2))
print '      Average Precision: {}'.format(round(outcome_df.precision.mean(), 2))
print '         Average Recall: {}'.format(round(outcome_df.recall.mean(), 2))
print '      Average F-1 Score: {}'.format(round(outcome_df.f1_score.mean(), 2))

plt.xkcd()
plt.figure(figsize=(10,5))
plt.plot(outcome_df.true_positive, label='True Positive')
plt.plot(outcome_df.true_negative, label='True Negative')
plt.plot(outcome_df.false_positive, label='False Positive')
plt.plot(outcome_df.false_negative, label='False Negative')
plt.title('100 Different Train Test Splits')
plt.xlabel('iteration')
plt.ylabel('count')
plt.legend()
# plt.savefig('graph.png', bbox_inches='tight', dpi=300)
plt.show()
