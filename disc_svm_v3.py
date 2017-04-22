#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
=====================
Classifier comparison
=====================

"""
print(__doc__)


# Code source:
#

import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
# from matplotlib.colors import ListedColormap
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification
from sklearn.svm import SVC
from svm_discrete import SVM_Milp
from svm_discrete import DILSVM
from sklearn.grid_search import GridSearchCV
# from sklearn.metrics import classification_report
from sklearn.datasets import fetch_mldata
# from sklearn.cross_validation import train_test_split
from sklearn import preprocessing
from numpy import array
from sklearn.datasets.california_housing import fetch_california_housing

le = preprocessing.LabelEncoder()

X_thyroid = pd.read_csv('mldata/Xthyroid.csv')
Y_thyroid = pd.read_csv('mldata/Ythyroid.csv')
Y_thyroid = Y_thyroid['x'].tolist()
le = preprocessing.LabelEncoder()
le.fit(Y_thyroid)
Y_thyroid = le.transform(Y_thyroid)
Y_thyroid = (2*(Y_thyroid-0.5)).astype(int)


thyroid = (X_thyroid, Y_thyroid)

X_car = pd.read_csv('mldata/Xcar.csv')
Y_car = pd.read_csv('mldata/Ycar.csv')
Y_car = array(Y_car)
ind = (Y_car == 'acc') | (Y_car == 'unacc')
ind1 = np.where(ind)

Y_car = Y_car[list(ind1[0])]
X_car = X_car.ix[list(ind1[0])]

# Y_car= Y_car['x'].tolist()
le = preprocessing.LabelEncoder()
le.fit(Y_car)
Y_car = le.transform(Y_car).T[0]
print(type(Y_car[1]))
Y_car = (2*(Y_car-0.5)).astype(int)

car = (X_car, Y_car)

X_spam = pd.read_csv('mldata/Xspam.csv')
Y_spam = pd.read_csv('mldata/Yspam.csv')
Y_spam = Y_spam['x'].tolist()
le = preprocessing.LabelEncoder()
le.fit(Y_spam)
Y_spam = le.transform(Y_spam)
print(type(Y_spam[1]))
Y_spam = (2*(Y_spam-0.5)).astype(int)
spam = (X_spam, Y_spam)


X_nursery = pd.read_csv('mldata/Xnursery.csv')
Y_nursery = pd.read_csv('mldata/Ynursery.csv')
ind1 = Y_nursery['x'].str.contains('very_recom')
Y_nursery[ind1] = 'others'

Y_nursery = Y_nursery['x'].tolist()
le = preprocessing.LabelEncoder()
le.fit(Y_nursery)
Y_nursery = le.transform(Y_nursery)
Y_nursery = (2*(Y_nursery-0.5)).astype(int)

nursery = (X_nursery, Y_nursery)
# print(Y_nursery.shape)

X_splice = pd.read_csv('mldata/Xsplice.csv')
Y_splice = pd.read_csv('mldata/Ysplice.csv')
ind_splice = Y_splice['x'].str.contains('EI|N')

X_splice = X_splice.ix[ind_splice]
Y_splice = Y_splice.ix[ind_splice]

Y_splice = Y_splice['x'].tolist()

le = preprocessing.LabelEncoder()
le.fit(Y_splice)
Y_splice = le.transform(Y_splice)
Y_splice = (2*(Y_splice - 0.5)).astype(int)

splice = (X_splice, Y_splice)


classifiers = [
    SVC(kernel="linear"),
    SVM_Milp(C=20),
    DILSVM()]

rng = np.random.RandomState(1)
test_data_home = '~/DiscreteClassif/github/'
cod_rna = fetch_mldata('cod-rna', data_home=test_data_home)

breast = fetch_mldata('breast-cancer', data_home=test_data_home)
X = cod_rna.data
y = cod_rna.target
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y, test_size=300)

X = X_test
y = y_test
codrna = (X, y)

X = breast.data
y = breast.target
y = y - 3
brc = (X, y)

X, y = make_classification(n_features=20, n_samples=1000,
                           n_redundant=0, n_informative=2,
                           random_state=1, n_clusters_per_class=1)
y = (-2*(y-0.5)).astype(int)
X += 2 * rng.uniform(size=X.shape)
linearly_separable = (X, y)

from milksets import german
X, y = german.load()

y = (-2*(y-0.5)).astype(int)
germ = (X, y)

column_names = ["gender", "length", "diameter", "height", "whole weight",
                "shucked weight", "viscera weight", "shell weight", "rings"]

data = pd.read_csv("mldata/abalone.data", names=column_names)
for label in "MFI":
    data[label] = data["gender"] == label
del data["gender"]
y = (data.rings.values > np.median(data.rings.values)).astype(int)
y = (-2*(y-0.5)).astype(int)
del data["rings"]  # remove rings from data, so we can convert
# all the dataframe to a numpy 2D array.
X = data.values.astype(np.float)

abalon = (X, y)

cal_housing = fetch_california_housing(data_home=test_data_home)

# split 80/20 train-test
X_calh = cal_housing.data
Y_calh = cal_housing.target

Y_calh = Y_calh > np.median(Y_calh)
# le = preprocessing.LabelEncoder()
# le.fit(Y_calh)
# Y_calh = le.transform(Y_calh)

Y_calh = Y_calh.astype(int)
Y_calh = (-2*(Y_calh-0.5)).astype(int)

calhous = (X_calh, Y_calh)

name_clf = ["Linear SVM", "SVM-DISC", "DILSVM"]

name_ds = ["breast-cancer", "nursery", "car", "cod-rna",
           "thyroid", "splice",  "spam", "calhous", "abalone",
           "german", "linear-seperable"]


datasets = [brc, nursery, car, codrna, thyroid,
            splice, spam, calhous, abalon,
            germ, linearly_separable]

dats = zip(name_ds, datasets)

# dats = dats[5:6]
ind_datasets = range(11)
# dat = [dats[i] for i in [0, 1, 2, 6]]
dat = [dats[i] for i in ind_datasets]
nm_dat = [name_ds[i] for i in ind_datasets]
ind_rnd = range(5)
resul = np.zeros((len(ind_rnd), len(dat), len(name_clf)))

print(type(dat))

param_grid1 = {'C': [2**i for i in xrange(-3, 11, 2)]}
param_grid2 = {'C': [2**i for i in xrange(-3, 11, 2)]}
param_grid3 = {'C': [2**i for i in xrange(-3, 11, 2)]}
# iterate over datasets
df = np.empty((len(dat), len(name_clf)), dtype=object)
for rnd in ind_rnd:
    print('random iteration:', rnd)
    for ind_ds, a in enumerate(dat):
        # preprocess dataset, split into training and test part
        nm_ds, ds = a
        print(nm_ds)
        X, y = ds
        print(X.shape)
        X = StandardScaler().fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.4)
        param_grd = [param_grid1, param_grid2, param_grid3]
        # iterate over classifiers
        for ind_clf, (nm_clf, clf, prm) in enumerate(zip(name_clf,
                                                         classifiers,
                                                         param_grd)):
            print((nm_ds, nm_clf))
            gs_cv = GridSearchCV(clf, prm, n_jobs=4, cv=2,
                                 scoring='accuracy',
                                 verbose=0).fit(X_train, y_train)
            clf1 = gs_cv.fit(X_train, y_train)
            clf1.fit(X_train, y_train)

            print('Best hyperparameters: %r' % gs_cv.best_params_['C'])
            score = clf1.score(X_test, y_test)
            resul[rnd][ind_ds][ind_clf] = score
            print("accuracy:", score)
    print (resul)
resul1 = resul.reshape(-1, 3)
df = pd.DataFrame(resul1, columns=["Linear_SVM", "SVM_DISC", "DILSVM"],
                  index=zip(nm_dat * len(ind_rnd), ind_rnd * len(nm_dat)))

print(df)
#    df3 = pd.DataFrame(df, columns=["Linear SVM", "SVM-DISC", "DILSVM"])
df.to_csv('compare_result_a3_C15.csv')
