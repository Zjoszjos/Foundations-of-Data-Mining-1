import openml as oml
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score, ShuffleSplit,train_test_split


oml.config.apikey = 'fa31aae3ceb0dba388acb20276cd75d3'
mnist_data = oml.datasets.get_dataset(554) # Download MNIST data
X, y = mnist_data.get_data(target=mnist_data.default_target_attribute); # Get the predictors X and the labels y


# Assignment 1.3
logReg = LogisticRegression()
SVC =  LinearSVC()
xReduced, _, yReduced, _= train_test_split(X, y, train_size=0.1, stratify=y)

#print("Logistic Regression Score: "+str(np.mean(cross_val_score(logReg, xReduced, yReduced, cv=10))))
#print("Linear SVC Score: "+str(np.mean(cross_val_score(SVC, xReduced, yReduced, cv=10))))

# #
# Logistic Regression Score: 0.823578260286
# Linear SVC Score: 0.838293082241
# #

for C, pen in [(x,y) for x in [0.0001, 0.001, 0.01] for y in ['l1','l2']]:
    logReg = LogisticRegression(C=C, penalty=pen, n_jobs=4)
    if(pen=='l1'):
        SVC = LinearSVC(C=C, penalty=pen, dual=False)
    else:
        SVC = LinearSVC(C=C, penalty=pen)
    print("Logistic Regression Score (C: " + str(C) + ", pen: " + pen + "): " + str(np.mean(cross_val_score(logReg, xReduced, yReduced, cv=10))))
    print("Linear SVC Score(C: " + str(C) + ", pen: " + pen + "): " + str(np.mean(cross_val_score(SVC, xReduced, yReduced, cv=10))))

# Logistic Regression Score (C: 0.0001, pen: l1): 0.841149138029
# Linear SVC Score(C: 0.0001, pen: l1): 0.879000672483
# Logistic Regression Score (C: 0.0001, pen: l2): 0.876996975084
# Linear SVC Score(C: 0.0001, pen: l2): 0.851140744184
# Logistic Regression Score (C: 0.001, pen: l1): 0.892145759666
# Linear SVC Score(C: 0.001, pen: l1): 0.893711894372
# Logistic Regression Score (C: 0.001, pen: l2): 0.857425305657
# Linear SVC Score(C: 0.001, pen: l2): 0.834853333378
# Logistic Regression Score (C: 0.01, pen: l1): 0.884980695318
# Linear SVC Score(C: 0.01, pen: l1): 0.863847424472
# Logistic Regression Score (C: 0.01, pen: l2): 0.841995451328
# Linear SVC Score(C: 0.01, pen: l2): 0.840416469404
# Logistic Regression Score (C: 0.01, pen: l1): 0.884430994756
# Linear SVC Score(C: 0.01, pen: l1): 0.86857464496
# Logistic Regression Score (C: 0.01, pen: l2): 0.849284579068
# Linear SVC Score(C: 0.01, pen: l2): 0.845999417135
# Logistic Regression Score (C: 0.1, pen: l1): 0.859850225402
# Linear SVC Score(C: 0.1, pen: l1): 0.849290283423
# Logistic Regression Score (C: 0.1, pen: l2): 0.838710581245
# Linear SVC Score(C: 0.1, pen: l2): 0.842566901221
# Logistic Regression Score (C: 1, pen: l1): 0.843988419018
# Linear SVC Score(C: 1, pen: l1): 0.842293896335
# Logistic Regression Score (C: 1, pen: l2): 0.835428527671
# Linear SVC Score(C: 1, pen: l2): 0.845438586573
# Logistic Regression Score (C: 100, pen: l1): 0.833580298594
# Linear SVC Score(C: 100, pen: l1): 0.842993692039
# Logistic Regression Score (C: 100, pen: l2): 0.833140092786
# Linear SVC Score(C: 100, pen: l2): 0.845432052166
