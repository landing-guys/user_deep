# -*- coding: utf-8 -*-
"""
Created on Sun Oct 11 20:43:58 2020

@author: Jerry
"""


import pandas as pd
import numpy as np
from math import ceil
from math import floor
from collections import Counter
from surprise import Dataset
from surprise import Reader
from surprise import KNNWithMeans
from surprise import accuracy
from surprise import SVD
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


'''
path:数据的路径
no_of_criteria:准则的数量
k_item:数据筛选时，筛选项目的阈值（一个项目至少被k_item个用户评论过）
k_user:数据筛选时，筛选项目的阈值（一个用户至少评论过k_user个项目）
random_state:划分数据集时的随机种子
'''
colnames = ['id','uid','iid','total'] + ['c'+str(i+1) for i in range(5)]
df = pd.read_csv('./Datas/alldata-188854.txt', sep='\t', names = colnames)
c_uid = Counter(df.uid).most_common()
c_iid = Counter(df.iid).most_common()
iids, uids = [], []
for iid in c_iid:
    if iid[1] >= 10:
        iids.append(iid[0])
for uid in c_uid:
    if uid[1] >=3:
        uids.append(uid[0])
df1 = df[df['iid'].isin(iids)]
df2 = df1[df1['uid'].isin(uids)]
df3 = df2.reset_index(drop=True)
n_ratings = len(df3)
test_size, train_size = validate_train_test_sizes(n_ratings)
random_state = 666
rng = np.random.RandomState(random_state)
permutation = rng.permutation(n_ratings)
trainset = permutation[:test_size]
testset = permutation[test_size:(test_size + train_size)]
train = df3[df3.index.isin(trainset)]
test = df3[df3.index.isin(testset)]
c_uid_train = Counter(train.uid).most_common()
c_uid_test = Counter(test.uid).most_common()
# uid_train, uid_test = [], []
# for c in c_uid_train:
#     uid_train.append(c[0])
# for c in c_uid_test:
#     uid_test.append(c[0])

r = Reader(rating_scale=(1,5))
options = {'name': 'pearson', 'user_based': False}
total_train = Dataset.load_from_df(train[['uid','iid','total']], reader = r).build_full_trainset()
total_algo = KNNWithMeans(sim_options = options).fit(total_train)
total_test = np.array(test[['uid','iid','total']])
total_p = total_algo.test(total_test)
accuracy.mae(total_p)
#小度还是大度用户指的是训练集中的数据（）

