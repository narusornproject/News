import preprocessing as pre
import numpy
from sklearn.model_selection import KFold
import json
import algorithm_ml as al
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


def train_test(corpus, bagWord, train, test, type):
    acc = 0
    for x in test:
        maxValue = 0
        predict = createObj('', '', '', '', '', '')
        for y in train:
            if type == 'cosine':
                try: cosine = al.Cosine(sims[pre.Fill(x, len(corpus))], sims[pre.Fill(y, len(corpus))])
                except: cosine = 0
                if cosine > maxValue:
                    maxValue = cosine
                    predict = createObj(corpus.index(x), type, cosine,
                                        dl[corpus.index(y)]['CategoryThai'],
                                        dl[corpus.index(x)]['CategoryThai'],
                                        dl[corpus.index(y)]['CategoryThai'] == dl[corpus.index(x)]['CategoryThai'])

            else:
                try: jaccard = al.jaccard(set(x), set(y))
                except: jaccard = 0
                if jaccard > maxValue:
                    maxValue = jaccard
                    predict = createObj(bagWord.index(x), type, jaccard,
                                        dl[bagWord.index(y)]['CategoryThai'],
                                        dl[bagWord.index(x)]['CategoryThai'],
                                        dl[bagWord.index(y)]['CategoryThai'] == dl[bagWord.index(x)]['CategoryThai'])

        print(predict)
        data.append(predict)
        if predict['ck']:
            acc += 1
    accuracy_set = str((acc / len(test)) * 100)
    print('Accuracy(%): ', accuracy_set)


if __name__ == '__main__':
    data = []
    df, dl = pre.GetNews()
    bagWord = pre.createBag(df)
    dictionary, tf_idf, corpus, sims = pre.createTFIDF(bagWord)

    """ Process reduce feature"""
    # min_count = 1
    # index_reduce, vocab_tf = rf.reduceFeature(dictionary, corpus, min_count)
    # del dictionary, tf_idf, corpus, sims
    # bagWord_reduce = rf.wordReduce(bagWord, index_reduce)
    # dictionary, tf_idf, corpus, sims = pre.createTFIDF(bagWord_reduce)

    x = numpy.array(corpus)
    y = numpy.array(bagWord)
    kf = KFold(n_splits=10)
    kf.get_n_splits(x)

    for train_index, test_index in kf.split(x):
        print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        train_test(corpus, bagWord, X_train, X_test, 'cosine')
        train_test(corpus, bagWord, y_train, y_test, 'jaccard')

    with open('data.json', 'a+', encoding='utf-8') as outfile:
        json.dump(data, outfile, ensure_ascii=False)
