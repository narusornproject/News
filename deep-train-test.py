from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score
import warnings
import numpy as np
from sklearn.model_selection import KFold
import algorithm_ml as ml
import algorithm_fs as fs


# decisionTree()
# rbfnn()
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
genetic_data, features, labels = fs.readCsv('Data/chi-100')

# X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.25, random_state=0)
#
# names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Gaussian Process",
#          "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
#          "Naive Bayes", "QDA"
#         ]
#
# classifiers = [
#     KNeighborsClassifier(),
#     SVC(kernel="linear"),
#     SVC(kernel="rbf"),
#     GaussianProcessClassifier(),
#     DecisionTreeClassifier(),
#     RandomForestClassifier(),
#     MLPClassifier(),
#     AdaBoostClassifier(),
#     GaussianNB(),
#     QuadraticDiscriminantAnalysis()
# ]
#
# # iterate over classifiers
# results = {}
# for name, clf in zip(names, classifiers):
#     scores = cross_val_score(clf, X_train, y_train, cv=10)
#     results[name] = scores
#
# for name, scores in results.items():
#     print("%20s | Accuracy: %0.2f%% (+/- %0.2f%%)" % (name, 100*scores.mean(), 100*scores.std() * 2))

x = np.array(features)
y = np.array(labels)

kf = KFold(n_splits=10)
kf.get_n_splits(x)
round = 1
for train_index, test_index in kf.split(x):
    # print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = x[train_index], x[test_index]
    y_train, y_test = y[train_index], y[test_index]
    if round in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:

        # print(y_test)

        clf = ml.SVM(X_train, y_train, 'onevsone')
        y_pred = clf.predict(X_test)
        # print(y_pred)
        print('Accuracy : %.2f%%' % (accuracy_score(y_test, y_pred)*100))

        # clf = ml.SVM(X_train, y_train, 'onevsrest')
        # y_pred = clf.predict(X_test)
        # print(y_pred)
        # print('Accuracy : %.2f%%' % (accuracy_score(y_test, y_pred) * 100))

        # gp = ml.rbfnn(X_train, y_train)
        # y_pred = gp.predict(X_test)
        # print(y_test)
        # print(y_pred)
        # print('Accuracy : %.2f%%' % (accuracy_score(y_test, y_pred) * 100))

        # tree = ml.decisionTree(X_train, y_train)
        # y_pred = tree.predict(X_test)
        # print(y_test)
        # print(y_pred)
        # print('Accuracy : %.2f%%' % (accuracy_score(y_test, y_pred) * 100))
        #
        # print('--------------------------------------------------')