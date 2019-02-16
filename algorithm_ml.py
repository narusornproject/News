from scipy import spatial


def jaccard(a, b):
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))


def Cosine(dataSetI, dataSetII):
    result = 1 - spatial.distance.cosine(dataSetI, dataSetII)
    # print(result)
    return result
