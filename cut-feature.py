import preprocessing as pre
import pandas as pd
# from tabulate import tabulate

data = []
df, dl = pre.GetNews()
bagWord = pre.createBag(df)
dictionary, tf_idf, corpus, sims = pre.createTFIDF(bagWord)

model = pre.createModel(bagWord)


def reduceFeature(corpus):
    vocab_tf = [dict(i) for i in corpus]
    vocab_tf = list(pd.DataFrame(vocab_tf).sum(axis=0))

    index_reduce = indexReduce(vocab_tf, 1)
    corpusNew = removeIndex(corpus, index_reduce)
    return index_reduce, corpusNew


def indexReduce(vocab_tf, min_count):
    #  [num.index(num[i]) for i in range(0, len(num)) if not num[i] == 1 ]
    index_reduce = [vocab_tf.index(vocab_tf[i], i) for i in range(0, len(vocab_tf)) if vocab_tf[i] == min_count]
    return index_reduce


def removeIndex(corpus, index_reduce):
    # [corpus[i].remove(corpus[i][j]) for i in range(0, len(corpus)-1) for j in range(0, len(corpus[i])-1) if not corpus[i][j][0] in index_reduce]
    corpusNew = []
    for i in range(0, len(corpus)):
        doc = []
        for j in range(0, len(corpus[i])):
            if not corpus[i][j][0] in index_reduce:
                doc.append(corpus[i][j])
                # corpus[i].remove(corpus[i][j])
        corpusNew.append(doc)
    return corpusNew


print(corpus)
index_reduce, corpusNew = reduceFeature(corpus)
print(index_reduce)
print(corpusNew)
