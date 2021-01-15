# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 08:33:06 2020

@author: Jerry
"""


import numpy as np
import pandas as pd
from surprise import Dataset
from surprise import Reader
from surprise import KNNWithMeans
from surprise import accuracy
from surprise import SVD
from surprise import SlopeOne
from math import ceil
from math import floor
from math import log2
from collections import Counter
from surprise.model_selection import train_test_split
from surprise.prediction_algorithms import predictions
from sklearn.linear_model import LinearRegression


def validate_train_test_sizes(n_ratings, test_size = 0.2, train_size = None):
    #
    if test_size is not None and test_size >= n_ratings:
        raise ValueError('test_size={0} should be less than the number of '
                             'ratings {1}'.format(test_size, n_ratings))
    if train_size is not None and train_size >= n_ratings:
        raise ValueError('train_size={0} should be less than the number of'
                             ' ratings {1}'.format(train_size, n_ratings))

    if np.asarray(test_size).dtype.kind == 'f':
        test_size = ceil(test_size * n_ratings)
    if train_size is None:
        train_size = n_ratings - test_size
    elif np.asarray(train_size).dtype.kind == 'f':
        train_size = floor(train_size * n_ratings)

    if test_size is None:
        test_size = n_ratings - train_size

    if train_size + test_size > n_ratings:
        raise ValueError('The sum of train_size and test_size ({0}) '
                             'should be smaller than the number of '
                             'ratings {1}.'.format(train_size + test_size,
                                                   n_ratings))
    return int(train_size), int(test_size)

no_of_criteria = 5
split_options = {'k_item': 10, 'k_user':5, 'random_state':6}
read_path = './Datas/yahoo.txt'
colnames = ['id','uid','iid','total'] + ['c'+str(i+1) for i in range(no_of_criteria)]
df = pd.read_csv(read_path, sep = '\t', names = colnames)
c_uid = Counter(df.uid).most_common()
c_iid = Counter(df.iid).most_common()
iids, uids = [], []
for iid in c_iid:
    if iid[1] >= split_options['k_item']:
        iids.append(iid[0])
for uid in c_uid:
    if uid[1] >= split_options['k_user']:
        uids.append(uid[0])
df1 = df[df['iid'].isin(iids)]
df2 = df1[df1['uid'].isin(uids)]
df3 = df2.reset_index(drop=True)
n_ratings = len(df3)
test_size, train_size = validate_train_test_sizes(n_ratings)
random_state = split_options['random_state']
rng = np.random.RandomState(random_state)
permutation = rng.permutation(n_ratings)
trainset = permutation[:test_size]
testset = permutation[test_size:(test_size + train_size)]

trainDatas = df3[df3.index.isin(trainset)].sort_values(by='uid')
testDatas = df3[df3.index.isin(testset)].sort_values(by='uid')

algos = []
df = trainDatas
names = locals()
r = Reader(rating_scale=(1,5))
# 读取、划分数据;训练预测数据
total = Dataset.load_from_df(df[['uid','iid','total']], reader = r)
total_train = total.build_full_trainset()
sim_options={'name': 'pearson',
               'user_based': True  # compute  similarities between items
               }
k=300
total_algo = KNNWithMeans(k=k, sim_options = sim_options)
total_algo.fit(total_train)
algos.append(total_algo)
for i in range(1, no_of_criteria+1):
    names['c' + str(i)] = Dataset.load_from_df(df[['uid','iid','c'+str(i)]], reader = r)
    names['c' + str(i) + '_train'] = names.get('c' + str(i)).build_full_trainset()
    names['algo_c' + str(i)] = KNNWithMeans(k=k, sim_options = sim_options)
    names.get('algo_c' + str(i)).fit(names.get('c'+str(i)+'_train'))
    algos.append(names.get('algo_c' + str(i)))

r = Reader(rating_scale=(1,5))
df = testDatas
total_test = np.array(df[['uid','iid','total']])
total_p = algos[0].test(total_test)
for i in range(1, no_of_criteria+1):
    # names['c' + str(i) + '_test'] = np.array(df[['uid','iid', 'c' + str(i)]])
    names['c' + str(i) + '_test'] = np.array(df[['uid','iid', 'c' + str(i)]])
    names['c' + str(i) + '_p'] = algos[i].test(names.get('c' + str(i) + '_test'))

x = trainDatas[['c' + str(i) for i in range(1, no_of_criteria+1)]]
y = trainDatas[['total']]
x = np.array(x)
y = np.array(y)                
reg = LinearRegression()
reg.fit(x, y)
k = reg.coef_[0]
b = reg.intercept_[0]



