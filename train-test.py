from nltk import ConfusionMatrix
from sklearn.metrics import accuracy_score

import preprocessing as pre
import algorithm_ml as al
from numpy import array
from sklearn.model_selection import KFold
import json
import genetic_data as gd
import algorithm_fs as fs
import reduce_feature as rf
# from tabulate import tabulate


def createObj(doc, similarity, value, type_true, type_pre, ck):
    predict = {
        'doc': doc,
        'similarity': similarity,
        'value': value,
        'type_true': type_true,
        'type_pre': type_pre,
        'ck': ck
    }
    return predict


def train_test(train, test, type):
    type_news = ['อาชญากรรม', 'การเมือง', 'กีฬา', 'บันเทิง', 'การศึกษา', 'ต่างประเทศ']
    acc = 0
    x_actu = []
    x_pred = []
    for x in test:
        # print(sims[pre.Fill(x, len(corpus))])
        maxValue = 0
        if type == 'cosine': predict = createObj(corpus.index(x), type, '', '', '', '')
        else: predict = createObj(bagWord.index(x), type, '', '', '', '')
        for y in train:
            if type == 'cosine':
                try: cosine = al.Cosine(pre.convertTuple(pre.Fill(x, len(dictionary))), pre.convertTuple(pre.Fill(y, len(dictionary))))
                except: cosine = 0
                if cosine > maxValue:
                    maxValue = cosine
                    predict = createObj(corpus.index(x), type, cosine,
                                        dl[corpus.index(y)]['CategoryThai'],
                                        dl[corpus.index(x)]['CategoryThai'],
                                        dl[corpus.index(y)]['CategoryThai'] == dl[corpus.index(x)]['CategoryThai'])

            elif type == 'jaccard':
                try: jaccard = al.jaccard(set(x), set(y))
                except: jaccard = 0
                if jaccard > maxValue:
                    maxValue = jaccard
                    predict = createObj(bagWord.index(x), type, jaccard,
                                        dl[bagWord.index(y)]['CategoryThai'],
                                        dl[bagWord.index(x)]['CategoryThai'],
                                        dl[bagWord.index(y)]['CategoryThai'] == dl[bagWord.index(x)]['CategoryThai'])
        # print(predict)
        # if type == 'cosine': class_1.append(predict['value'])
        # else: class_2.append(predict['value'])
        try:
           x_actu.append(type_news.index(predict['type_true']))
           x_pred.append(type_news.index(predict['type_pre']))
        except:
           pass
        # data.append(predict)
        if predict['ck']:
            acc += 1
    # accuracy_set = str((acc / len(test)) * 100)
    # print('Accuracy(%): ', accuracy_set)
    return x_actu, x_pred


if __name__ == '__main__':
    df, dl = pre.GetNews()
    bagWord = pre.createBag(df)
    dictionary, tf_idf, corpus, sims = pre.createTFIDF(bagWord)

    # gd.createGenetic_data('originData', dictionary, corpus, dl)
    """ Process reduce feature"""

    # filename = 'Data/originData'
    # genetic_data, features, labels = fs.readCsv(filename)

    # res_mi = fs.mutual_information(features, labels)
    # res_chi = fs.chi_sqaure(features, labels)
    # res_rf = fs.reliefF(features, labels)
    # res_rg = fs.ratio_gain(filename)
    # res_ig = fs.ig.information_gain(features, labels)

    # index_reduce = fs.ranking(dictionary, res_ig, 100) # --> number of Reduce Feature
    # rf.wordReduce(bagWord, index_reduce)
    # del dictionary, tf_idf, corpus, sims
    # dictionary, tf_idf, corpus, sims = pre.createTFIDF(bagWord)
    #
    # gd.createGenetic_data('ig-100.csv', dictionary, corpus, dl)

    # rf.reduceFeature(dictionary, corpus, 0) # term frequency
    # rf.wordReduce(bagWord, index_reduce)
    # del dictionary, tf_idf, corpus, sims
    # dictionary, tf_idf, corpus, sims = pre.createTFIDF(bagWord)
    """ --------------------- """

    x = array(corpus)
    y = array(bagWord)
    kf = KFold(n_splits=10)
    kf.get_n_splits(x)

    round = 1
    for train_index, test_index in kf.split(x):
        data = []
        # print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]

        if round in [1, 2, 3, 4, 5, 6, 7, 9, 8, 10]:

            x_actu, x_pred = train_test(X_train, X_test, 'cosine')
            # cm = ConfusionMatrix(x_actu, x_pred)
            # print('%.2f%%' % (accuracy_score(x_actu, x_pred)*100))

            x_actu2, x_pred2 = train_test(y_train, y_test, 'jaccard')
            # cm = ConfusionMatrix(x_actu, x_pred)
            # print('Accuracy : %.2f%%' % (accuracy_score(x_actu2, x_pred2) * 100))

            with open('Data/train-test(' + str(round) + ').json', 'a+', encoding='utf-8') as outfile:
                json.dump(data, outfile, ensure_ascii=False)

        round += 1
