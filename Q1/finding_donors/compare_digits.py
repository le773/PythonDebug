# print '1'
from __future__ import division
from sklearn.grid_search import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn import datasets, cross_validation
from sklearn.neighbors import KNeighborsClassifier
from pprint import pprint


digits = datasets.load_digits()

kn = KNeighborsClassifier()
parameters = {
    'n_neighbors': [1, 2, 4, 6, 8, 10]
}

clf = GridSearchCV(kn, parameters)
clf.fit(digits.data, digits.target)

pprint(clf.grid_scores_)
