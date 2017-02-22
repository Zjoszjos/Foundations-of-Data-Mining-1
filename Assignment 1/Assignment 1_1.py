import openml as oml
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, ShuffleSplit,train_test_split


oml.config.apikey = 'fa31aae3ceb0dba388acb20276cd75d3'
mnist_data = oml.datasets.get_dataset(554) # Download MNIST data
X, y = mnist_data.get_data(target=mnist_data.default_target_attribute); # Get the predictors X and the labels y

#Assignment 1.1
# #
xTrain, xTest = np.split(X,[60000])
yTrain, yTest = np.split(y,[60000])

print("Data split into Test and Train")

knn = KNeighborsClassifier()
knn.fit(xTrain, yTrain)
print("KNN Trained")
count = 0
for i in range(0,10000):
    pred = knn.predict(xTest[i].reshape(1,-1))
    if(pred != yTest[i]) and count < 5:
        plt.imshow(xTest[i].reshape(28, 28), cmap=plt.cm.gray_r)
        plt.title("Predicted Number: " + str(pred[0]) + ", Actual Number: " + str(yTest[i]))
        count += 1
    elif count >= 5:
        plt.show()
        break
    else:
        continue
# #

