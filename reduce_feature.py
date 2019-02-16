import pandas as pd


def reduceFeature(dictionary, corpus, min_count):
    vocab_tf = [dict(i) for i in corpus]
    vocab_tf = list(pd.DataFrame(vocab_tf).sum(axis=0))

    index_reduce = indexReduce(dictionary, vocab_tf, min_count)
    # corpusNew = removeIndex(corpus, index_reduce)
    return index_reduce, vocab_tf


def indexReduce(dictionary, vocab_tf, min_count):
    #  [num.index(num[i]) for i in range(0, len(num)) if not num[i] == 1 ]
    index_reduce = [dictionary[vocab_tf.index(vocab_tf[i], i)] for i in range(0, len(vocab_tf)) if vocab_tf[i] == min_count]
    return index_reduce


def wordReduce(bagWord, index_reduce):
    for i in bagWord:
        for j in index_reduce:
            try: i.remove(j)
            except: pass
    return bagWord
