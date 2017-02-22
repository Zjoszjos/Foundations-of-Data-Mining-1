import openml as oml
from sklearn import ensemble
from sklearn.model_selection import ShuffleSplit, train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import numpy as np
import warnings
import pandas as pd

oml.config.apikey = 'fa31aae3ceb0dba388acb20276cd75d3'
eeg = oml.datasets.get_dataset(1471)
X, y = eeg.get_data(target=eeg.default_target_attribute)

# Out of bag errors can be retrieved from the RandomForest classifier.
# You'll need to loop over the number of trees.
# http://scikit-learn.org/stable/auto_examples/ensemble/plot_ensemble_oob.html

scale = [2 ** i for i in range(0,10)]

xTrain,xTest,yTrain,yTest = train_test_split(X,y, test_size=.1, random_state=0)
def calcBiasVariance(clf):
    # Bootstraps
    n_repeat = 100
    shuffle_split = ShuffleSplit(test_size=0.33, n_splits=n_repeat)

    # Store sample predictions
    y_all_pred = [[] for _ in range(len(y))]

    # Train classifier on each bootstrap and score predictions
    for i, (train_index, test_index) in enumerate(shuffle_split.split(X)):
        # Train and predict
        clf.fit(X[train_index], y[train_index])
        y_pred = clf.predict(X[test_index])

        # Store predictions
        for i, index in enumerate(test_index):
            y_all_pred[index].append(y_pred[i])

    # Compute bias, variance, error
    bias_sq = sum([(1 - x.count(y[i]) / len(x)) ** 2 * len(x) / n_repeat
                   for i, x in enumerate(y_all_pred)])
    var = sum([((1 - ((x.count(0) / len(x)) ** 2 + (x.count(1) / len(x)) ** 2)) / 2) * len(x) / n_repeat
               for i, x in enumerate(y_all_pred)])
    error = sum([(1 - x.count(y[i]) / len(x)) * len(x) / n_repeat
                 for i, x in enumerate(y_all_pred)])
    return bias_sq, var, error

warnings.filterwarnings('ignore')
errors = []
dtc = DecisionTreeClassifier(random_state=0)
dtc.fit(xTrain,yTrain)
tree_auc = roc_auc_score(yTest, dtc.predict(xTest))
val = calcBiasVariance(dtc)
print("For Decision Tree, Total Bias Squared: %.2f, Total Variance: %.2f, AUC %.2f"%(val[0], val[1], tree_auc))

for est in scale:
    clf = ensemble.RandomForestClassifier(n_estimators=int(est), n_jobs=4, oob_score=True, random_state=0)
    clf.fit(xTrain, yTrain)
    # error = (1-clf.oob_score_)
    # errors.append(error)

    #errors.append(1-np.mean(cross_val_score(clf,X, y, cv=10)))

    vals = calcBiasVariance(clf)

    for_auc = roc_auc_score(yTest, clf.predict(xTest))

    print("For %.0f estimators and Random Forest, Total Bias Squared: %.2f, Total Variance: %.2f, AUC: %.2f"%(est, vals[0], vals[1], for_auc))
plt.plot(scale, errors)
plt.xscale('log')
plt.show()


