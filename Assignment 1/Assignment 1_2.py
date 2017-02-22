import openml as oml
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, ShuffleSplit,train_test_split


oml.config.apikey = 'fa31aae3ceb0dba388acb20276cd75d3'
mnist_data = oml.datasets.get_dataset(554) # Download MNIST data
X, y = mnist_data.get_data(target=mnist_data.default_target_attribute); # Get the predictors X and the labels y


# # Assignment 1.2
# xReduced, temp = np.split(X, [6000])
# yReduced, temp = np.split(y, [6000])
# scores = []
# knn = KNeighborsClassifier()
# for k in range(1,50):
#     knn.n_neighbors=k
#     print("Calculating with k="+str(k))
#     scores.append(np.mean(cross_val_score(knn, xReduced, yReduced, cv=10)))
# x = range(1,50)
# plt.plot(x, scores, marker="o")
# plt.grid()
# plt.show()

# # Assignment 1.2.2
xReduced,_, yReduced, _= train_test_split(X, y, train_size=0.1, stratify=y)


def main():
    if __name__ == '__main__':
        scores = []
        knn = KNeighborsClassifier()
        for k in range(1,50):
            knn.n_neighbors=k
            print("Calculating with k="+str(k))
            mean = 0
            supSplit = ShuffleSplit(test_size=0.66, train_size=0.34, n_splits=100)
            mean += np.mean((cross_val_score(knn, xReduced, yReduced, cv=supSplit,n_jobs=4)))
            print(mean)
            scores.append(mean)
        x = range(1,50)
        plt.plot(x, scores, marker="o")
        plt.grid()
        plt.show()
        return
main()