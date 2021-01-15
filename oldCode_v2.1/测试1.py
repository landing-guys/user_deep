# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 21:08:33 2020

@author: Jerry
"""

import numpy as np
import pandas as pd
from surprise import Reader
from surprise import Dataset
from surprise import SVD
from sklearn.linear_model import LinearRegression
from surprise.prediction_algorithms import predictions
from surprise.model_selection import train_test_split



read_path = './yahoodata(step)/1/min/all.txt'
colnames = ['id','uid','iid','total'] + ['c'+str(i+1) for i in range(5)]
df = pd.read_csv(read_path, sep = '\t', names = colnames)
names = locals()
r = Reader(rating_scale=(1,5))
# 读取、划分数据;训练预测数据
total = Dataset.load_from_df(df[['uid','iid','total']], reader = r)
total_train, total_test = train_test_split(total, random_state = 3)
total_algo = SVD(n_factors = 20, n_epochs = 20,verbose = True)
total_algo.fit(total_train)
total_p = total_algo.test(total_test)
for i in range(1, 6):
    names['c' + str(i)] = Dataset.load_from_df(df[['uid','iid','c'+str(i)]], reader = r)
    names['c' + str(i) + '_train'], names['c' + str(i) + '_test'] = train_test_split(names.get('c' + str(i)), random_state = 3)
    names['algo_c' + str(i)] = SVD(n_factors = 20, n_epochs = 20, verbose = True)
    names.get('algo_c' + str(i)).fit(names.get('c'+str(i)+'_train'))
    names['c' + str(i) + '_p'] = names.get('algo_c' + str(i)).test(names.get('c'+str(i)+'_test'))
    #
multi_p = []
for i in range(1, 6):
    names['x'+str(i)] = np.array(names.get('c'+str(i)+'_train').build_testset())[:,2]
    if i == 1:
        x = names.get('x'+str(i))
    else:
        x = np.vstack((x,names.get('x'+str(i))))
x = x.T
y = np.array(total_train.build_testset())[:,2]
reg = LinearRegression()
reg.fit(x, y)
k = reg.coef_
b = reg.intercept_
for i in range(len(total_p)):
    s = 0
    for j in range(5):
        s = s + k[j]*names.get('c'+str(j+1)+'_p')[i].est
    s = s +b
    p = predictions.Prediction(total_p[i].uid, total_p[i].iid, total_p[i].r_ui, s, total_p[i].details)
    multi_p.append(p)
