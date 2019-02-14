import preprocessing as pre
import numpy
from sklearn.model_selection import KFold
import json
# from tabulate import tabulate

data = []
df, dl = pre.GetNews()
bagWord = pre.createBag(df)
dictionary, tf_idf, corpus, sims = pre.createTFIDF(bagWord)

x = numpy.array(corpus)
y = numpy.array(bagWord)
kf = KFold(n_splits=10)
kf.get_n_splits(x)


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
	acc = 0
	for x in test:
		maxValue = 0
		predict = createObj('', '', '', '', '', '')
		for y in train:
			if type == 'cosine':
				cosine = pre.Cosine(sims[pre.Fill(x, len(corpus))], sims[pre.Fill(y, len(corpus))])
				if cosine > maxValue:
					maxValue = cosine
					predict = createObj(corpus.index(x), type, cosine,
										dl[corpus.index(y)]['CategoryThai'],
										dl[corpus.index(x)]['CategoryThai'],
										dl[corpus.index(y)]['CategoryThai'] == dl[corpus.index(x)]['CategoryThai'])

			else:
				jaccard = pre.jaccard(set(x), set(y))
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


for train_index, test_index in kf.split(x):
	print("TRAIN:", train_index, "TEST:", test_index)
	X_train, X_test = x[train_index], x[test_index]
	y_train, y_test = y[train_index], y[test_index]
	train_test(X_train, X_test, 'cosine')
	train_test(y_train, y_test, 'jaccard')

with open('data.json', 'a+', encoding='utf-8') as outfile:
	json.dump(data, outfile, ensure_ascii=False)