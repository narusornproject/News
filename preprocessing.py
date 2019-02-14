import deepcut
import gensim
import numpy
import pymongo
from gensim.models import Word2Vec
from pythainlp.corpus import stopwords
import pandas as pd
from scipy import spatial
from sklearn.model_selection import KFold
# from tabulate import tabulate
import re
import string
import json


def GetNews():
    client = pymongo.MongoClient("mongodb://localhost:27017/")
    db = client['News']
    col = db['DNews']
    dataFrame = pd.DataFrame(list(col.find({'$or': [{'CategoryThai': 'อาชญากรรม'},
                                                    {'CategoryThai': 'การเมือง'},
                                                    {'CategoryThai': 'กีฬา'},
                                                    {'CategoryThai': 'บันเทิง'},
                                                    {'CategoryThai': 'การศึกษา'},
                                                    {'CategoryThai': 'ต่างประเทศ'}]},
                                           {'Header': True, 'CategoryThai': True, '_id': False}).limit(100)))
    dataList = list(col.find({'$or': [{'CategoryThai': 'อาชญากรรม'},
                                      {'CategoryThai': 'การเมือง'},
                                      {'CategoryThai': 'กีฬา'},
                                      {'CategoryThai': 'บันเทิง'},
                                      {'CategoryThai': 'การศึกษา'},
                                      {'CategoryThai': 'ต่างประเทศ'}]},
                             {'Header': True, 'CategoryThai': True, '_id': False}).limit(100))
    client.close()
    return dataFrame, dataList


def CleanText(str):
    """ Remove number """
    clr_str = re.sub(r'\d+', '', str).strip().strip("' '")

    """ Remove symbols """
    exclude = set(string.punctuation)
    clr_str1 = ''.join(ch for ch in clr_str if ch not in exclude)

    """ Remove whitespaces """
    exclude = set(string.whitespace)
    clr_str2 = ''.join(ch for ch in clr_str1 if ch not in exclude)
    return clr_str2


def CutStopWord(strArray):
    """ Remove stop words """
    stop_words = set(stopwords.words('thai'))
    result = [i for i in strArray if not i in stop_words]
    return result


def createBag(dataFrame):
    bagWord = []
    for x in dataFrame['Header']:
        new_str = CleanText(x)
        """ Cut words """
        bagWord.append(CutStopWord(deepcut.tokenize(new_str)))
    return bagWord


def createModel(bagWord):
    model = Word2Vec(bagWord, min_count=1)
    return model


def createTFIDF(bagWord):
    """ 1 """
    gen_docs = bagWord
    dictionary = gensim.corpora.Dictionary(gen_docs)

    # print(dictionary)
    # print("Number of words in dictionary:", len(dictionary))

    # for i in range(len(dictionary)):
    #     print(i, dictionary[i])

    corpus = [dictionary.doc2bow(gen_doc) for gen_doc in gen_docs]
    # print(corpus)

    tf_idf = gensim.models.TfidfModel(corpus)
    # print(tf_idf)

    s = 0
    for i in corpus:
        s += len(i)
    # print(s)

    # similarities gensim
    sims = gensim.similarities.Similarity('', tf_idf[corpus],
                                           num_features=len(dictionary))

    return dictionary, tf_idf, corpus, sims


def Cosine(dataSetI, dataSetII):
    result = 1 - spatial.distance.cosine(dataSetI, dataSetII)
    # print(result)
    return result


def Query(str, dictionary, tf_idf):
    query_doc = [w for w in CutStopWord(
        deepcut.tokenize(CleanText(str)))]
    query_doc_bow = dictionary.doc2bow(query_doc)
    query_doc_tf_idf = tf_idf[query_doc_bow]
    return query_doc_tf_idf


def value_sort(t):
    return t[0]


def Fill(tfidf, len_corpus):
    num = 0
    for x in tfidf:

        if x[0] > num:
            for y in range(num, x[0]):
                if num == x[0]:
                    break
                tfidf.append((int(y), 0))
                num = y
        num += 2
    for x in range(len(tfidf), len_corpus):
        tfidf.append((int(x), 0))
    tfidf.sort(key=value_sort)
    tfidfSort = tfidf
    return tfidfSort


data = []
df, dl = GetNews()
# print(tabulate(df, headers='keys', tablefmt='rst'))
bagWord = createBag(df)
# print(bagWord)

dictionary, tf_idf, corpus, sims = createTFIDF(bagWord)
# print(dictionary)
# print(tf_idf)
# print(corpus)

x = numpy.array(corpus)
y = numpy.array(dictionary)
kf = KFold(n_splits=10)
kf.get_n_splits(x)
round = 1
for train_index, test_index in kf.split(x):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = x[train_index], x[test_index]
    y_train, y_test = y[train_index], y[test_index]

    acc = 0
    for xtest in X_test:
        predict = {
            'no': 0,
            'cosine': 0,
            'type_true': '',
            'type_pre': '',
            'check': 0
        }
        # print('=================== Predict ===================')
        for xtrain in X_train:

            # print('train: ' + str(corpus.index(xtrain)) + ' test: ' + str(corpus.index(xtest)))
            # print(xtest + '||' + xtrain)
            try:
                cosine = Cosine(sims[Fill(xtrain, len(corpus))], sims[Fill(xtest, len(corpus))])
            except Exception as error:
                print(error)
                cosine = 0
                pass
            if cosine > predict['cosine']:
                predict['no'] = corpus.index(xtest)
                predict['cosine'] = cosine
                predict['type_true'] = dl[corpus.index(xtrain)]['CategoryThai']
                predict['type_pre'] = dl[corpus.index(xtest)]['CategoryThai']
                if predict['type_true'] == predict['type_pre']:
                    predict['check'] = 1
        """ log """
        data.append(predict)
        print(predict)
        if predict['check'] == 1:
            acc += 1
        # print('===============================================\n')
        # accuracy_set = str((acc/len(X_test)) * 100)
        # print('train: {} test: {}'.format(len(X_train), len(X_test)))
        # print('=================== Results ===================')
        # print(acc)
        # print(len(X_test))
        # print(100)
        # print('Round: ', round)
        # print('Accuracy: ', acc)
        # print('Accuracy(%): ', accuracy_set)
        # print('===============================================\n')
        # round += 1
    """ Output """
    accuracy_set = str((acc / len(X_test)) * 100)
    print('Accuracy(%): ', accuracy_set)
with open('data.json', 'a+', encoding='utf-8') as outfile:
    json.dump(data, outfile, ensure_ascii=False)