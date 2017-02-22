from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import LinearSVR
from sklearn.model_selection import cross_val_score, GridSearchCV

import openml as oml
import numpy as np

oml.config.apikey = 'fa31aae3ceb0dba388acb20276cd75d3'
liver = oml.datasets.get_dataset(8) # Download Liver-disorders data
X, y = liver.get_data(target=liver.default_target_attribute)
if __name__ == '__main__':
    LinReg = LinearRegression(n_jobs=4)
    print("Linear Regression Score: " + str(np.mean(cross_val_score(LinReg, X,y, cv=10))))
    SVR = LinearSVR()
    LogReg = LogisticRegression()
    gridParam = {"C": [0.0001, 0.001, 0.01,0.1,1,10,100,1000], "penalty": ['l1','l2']}
    grid = GridSearchCV(LogisticRegression(), param_grid=gridParam, n_jobs=4)
    print(cross_val_score(grid, X, y, cv=5))

