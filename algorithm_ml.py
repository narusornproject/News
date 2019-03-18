from scipy import spatial
import numpy as np
from sklearn import tree
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from sklearn.model_selection import train_test_split, GridSearchCV
import algorithm_fs as fs
import graphviz


def jaccard(a, b):
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))


def Cosine(dataSetI, dataSetII):
    result = 1 - spatial.distance.cosine(dataSetI, dataSetII)
    # print(result)
    return result


def SVM(X, y, classifier):
    # Fit Support Vector Machine Classifier

    param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01, 0.001, 0.00001, 10]}

    clf_grid = GridSearchCV(SVC(), param_grid, verbose=1)

    # Train the classifier
    clf_grid.fit(X, y)

    if classifier == 'onevsrest':
        clf = OneVsRestClassifier(SVC(C=clf_grid.best_params_['C'], kernel='linear', gamma=clf_grid.best_params_['gamma']))
    else:
        clf = OneVsOneClassifier(SVC(C=clf_grid.best_params_['C'], kernel='linear', gamma=clf_grid.best_params_['gamma']))
    # clf = SVC(C=clf_grid.best_params_['C'], kernel='rbf', gamma=clf_grid.best_params_['gamma'], decision_function_shape='ovr')

    clf.fit(X, y)

    return clf


def decisionTree(X, y):
    model = tree.DecisionTreeClassifier(
        criterion='entropy')  # for classification, here you can change the algorithm as gini or entropy (information gain) by default it is gini
    # model = tree.DecisionTreeRegressor() for regression
    # Train the model using the training sets and check score
    model.fit(X, y)

    return model


def rbfnn(X, y):
    # kernel = ConstantKernel(0.1, (1e-23, 1e5)) * \
    #          RBF(0.1 * np.ones(X.shape[1]), (1e-23, 1e10)) + \
    #          WhiteKernel(0.1, (1e-23, 1e5))
    # kernel = 1.0 * RBF([1.0])
    kernel = RBF(length_scale=1.0, length_scale_bounds=(1e-1, 10.0))
    gp = GaussianProcessClassifier(kernel)

    gp.fit(X, y)

    return gp
