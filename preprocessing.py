import deepcut
import gensim
import pymongo
from gensim.models import Word2Vec
from pythainlp.corpus import stopwords
import pandas as pd
import re
import string


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
    dictionary = gensim.corpora.Dictionary(bagWord)

    # print(dictionary)
    # print("Number of words in dictionary:", len(dictionary))

    # for i in range(len(dictionary)):
    #     print(i, dictionary[i])

    corpus = [dictionary.doc2bow(gen_doc) for gen_doc in bagWord]
    # print(corpus)

    tf_idf = gensim.models.TfidfModel(corpus)
    # print(tf_idf)

    # similarities gensim
    sims = gensim.similarities.Similarity('', tf_idf[corpus],
                                           num_features=len(dictionary))

    return dictionary, tf_idf, corpus, sims


def Query(str, dictionary, tf_idf):
    query_doc = [w for w in CutStopWord(
        deepcut.tokenize(CleanText(str)))]
    print(query_doc)
    query_doc_bow = dictionary.doc2bow(query_doc)
    print(query_doc_bow)
    query_doc_tf_idf = tf_idf[query_doc_bow]
    print(query_doc_tf_idf)
    return query_doc_tf_idf


def value_sort(t):
    return t[0]


def Fill(doc, len_dictionary):
    result, last = [], 0.0
    for d in doc:
        while last < d[0]:
            result.append((last, 0))
            last += 1
        result.append(d)
        last = d[0] + 1
    if len(result) != len_dictionary:
        for a in range(len(result), len_dictionary):
            result.append((a, 0))
    return result
