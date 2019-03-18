import sklearn.feature_selection
import pandas as pd
from sklearn.linear_model import LogisticRegression
import numpy as np
import informationgain as ig
import warnings
import preprocessing as pre
import reduce_feature as rf


def mutual_information(features, labels):
    res = sklearn.feature_selection.mutual_info_classif(features, labels, discrete_features=True)
    return res


def ratio_gain(filename):
    genetic_data, features, labels = readCsv(filename)
    res = ig.information_gain(features, labels)
    total_entropy = [entropy(genetic_data[str(i)].tolist()) for i in range(len(genetic_data.keys())-1)]
    rg = res/total_entropy
    # total_entropy = entropy(labels)
    # rg = np.true_divide(res, total_entropy)
    return rg


def chi_sqaure(features, labels):
    ch2 = sklearn.feature_selection.SelectKBest(sklearn.feature_selection.chi2, "all")
    ch2.fit_transform(features, labels)
    res = ch2.scores_

    return res


def reliefF(features, labels):
    model = LogisticRegression()
    res = sklearn.feature_selection.RFE(model)
    res = res.fit(features, labels)

    return res.ranking_


def readCsv(filename):
    genetic_data = pd.read_csv(filename + '.csv', index_col=None)

    features, labels = genetic_data.drop('lables', axis=1).values, genetic_data['lables'].values
    return genetic_data, features, labels


def entropy(target_col):
    """
    Calculate the entropy of a dataset.
    The only parameter of this function is the target_col parameter which specifies the target column
    """
    elements, counts = np.unique(target_col, return_counts=True)
    entropy = np.sum([(-counts[i]/np.sum(counts))*np.log2(counts[i]/np.sum(counts)) for i in range(len(elements))])
    return entropy


def ranking(dictionary, res, num):
    index_reduce = []
    for score, name in sorted(zip(res, dictionary))[:num]:
        index_reduce.append(dictionary[name])
        # print(name, score)
    return index_reduce
