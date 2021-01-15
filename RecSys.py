# -*- coding: utf-8 -*-
"""
Created on Sat Sep 26 21:41:29 2020

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

'''
read_path：实验数据的路径
outcome_path：实验结果的保存路径
combin_func：聚合函数的方法，目前有：‘avg’，简单平均；‘total_reg’，整体回归模型
is_total：整体评分是否作为准则：‘0’，代表不是；‘1’，代表是
'''

class Experiment:
    
    
    def __init__(self, no_of_criteria=5):
        self.no_of_criteria = no_of_criteria


    @staticmethod
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
    
    
    def Info_Entropy(self):
        df = self.trainDatas.sort_values(by='uid')
        all_user = []
        c = Counter(df.uid).most_common()
        c.sort()
        for i in c:
            all_user.append(i[0])
        H = []
        names = locals()
        for uid in all_user:
            td = df[df['uid'].isin([uid])]
            h = [uid]
            if len(td) > 1:
                total_h = 0
                for i in range(5):
                    tc = Counter(td['c'+str(i+1)]).most_common()
                    t = np.array(tc)
                    p = t[:,1]/sum(t[:,1])
                    th = - sum([i*log2(i) for i in p])
                    total_h = total_h + th
                    h.append(th)
                h.append(total_h)
                if total_h == 0:
                    h = [uid]
                    for i in range(self.no_of_criteria):
                        h.append(1)
                    h.append(5)
            else:
                for i in range(self.no_of_criteria):
                    h.append(1)
                h.append(5)
            H.append(h)
        self.H = H
        
        
        
    def total_reg_param(self):
        x = self.trainDatas[['c' + str(i) for i in range(1, self.no_of_criteria+1)]]
        y = self.trainDatas[['total']]
        x = np.array(x)
        y = np.array(y)                
        reg = LinearRegression()
        reg.fit(x, y)
        self.k = reg.coef_[0]
        self.b = reg.intercept_[0]
        
    def read_split_Datas(self, read_path, split_options = {'k_item':10,'k_user':3,'random_state':666}):
        
        colnames = ['id','uid','iid','total'] + ['c'+str(i+1) for i in range(self.no_of_criteria)]
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
        test_size, train_size = self.validate_train_test_sizes(n_ratings)
        random_state = split_options['random_state']
        rng = np.random.RandomState(random_state)
        permutation = rng.permutation(n_ratings)
        trainset = permutation[:test_size]
        testset = permutation[test_size:(test_size + train_size)]
        self.trainDatas = df3[df3.index.isin(trainset)].sort_values(by='uid')
        self.testDatas = df3[df3.index.isin(testset)].sort_values(by='uid')
        self.total_reg_param()
        
    def splitTestDatas(self, k):
        self.more_test, self.min_test = [], []
        df = self.trainDatas
        df1 = self.testDatas
        c = Counter(df.uid).most_common()
        moreid = []
        for i in c:
            if i[1] > k:
                moreid.append(i[0])        
        self.more_test = df1[df1['uid'].isin(moreid)]        
        self.min_test = df1[~df1['uid'].isin(moreid)]
        
        
        
    def KNN_train(self, k = 20, options = {'name': 'pearson', 'user_based': False}):
        '''
        seed：int-3划分训练集测试集的随机种子
        k：int-40，最大邻居数量
        options：dict-{'name': 'pearson', 'user_based': False}，算法的选项，默认为Pearson相似度，基于项目的方法
        '''
        self.algos = []
        df = self.trainDatas
        names = locals()
        r = Reader(rating_scale=(1,5))
        # 读取、划分数据;训练预测数据
        total = Dataset.load_from_df(df[['uid','iid','total']], reader = r)
        total_train = total.build_full_trainset()
        total_algo = KNNWithMeans(k, sim_options = options)
        total_algo.fit(total_train)
        self.algos.append(total_algo)
        for i in range(1, self.no_of_criteria+1):
            names['c' + str(i)] = Dataset.load_from_df(df[['uid','iid','c'+str(i)]], reader = r)
            names['c' + str(i) + '_train'] = names.get('c' + str(i)).build_full_trainset()
            names['algo_c' + str(i)] = KNNWithMeans(k, sim_options = options)
            names.get('algo_c' + str(i)).fit(names.get('c'+str(i)+'_train'))
            self.algos.append(names.get('algo_c' + str(i)))
        

    def SVD_train(self, options = {'n_factors':50, 'n_epochs':25, 'random_state':666}):
        '''
        seed：int-3划分训练集测试集的随机种子
        k：int-40，最大邻居数量
        options：dict-{'name': 'pearson', 'user_based': False}，算法的选项，默认为Pearson相似度，基于项目的方法
        '''
        self.algos = []
        df = self.trainDatas
        names = locals()
        r = Reader(rating_scale=(1,5))
        # 读取、划分数据;训练预测数据
        total = Dataset.load_from_df(df[['uid','iid','total']], reader = r)
        total_train = total.build_full_trainset()
        total_algo = SVD(n_factors = options['n_factors'], n_epochs = options['n_epochs'], random_state = options['random_state'], verbose = True)
        total_algo.fit(total_train)
        self.algos.append(total_algo)
        for i in range(1, self.no_of_criteria+1):
            names['c' + str(i)] = Dataset.load_from_df(df[['uid','iid','c'+str(i)]], reader = r)
            names['c' + str(i) + '_train'] = names.get('c' + str(i)).build_full_trainset()
            names['algo_c' + str(i)] = SVD(n_factors = options['n_factors'], n_epochs = options['n_epochs'], verbose = True)
            names.get('algo_c' + str(i)).fit(names.get('c'+str(i)+'_train'))
            self.algos.append(names.get('algo_c' + str(i)))
            
    def SlopeOne_train(self):
        '''
        seed：int-3划分训练集测试集的随机种子
        k：int-40，最大邻居数量
        options：dict-{'name': 'pearson', 'user_based': False}，算法的选项，默认为Pearson相似度，基于项目的方法
        '''
        self.algos = []
        df = self.trainDatas
        names = locals()
        r = Reader(rating_scale=(1,5))
        # 读取、划分数据;训练预测数据
        total = Dataset.load_from_df(df[['uid','iid','total']], reader = r)
        total_train = total.build_full_trainset()
        total_algo = SlopeOne()
        total_algo.fit(total_train)
        self.algos.append(total_algo)
        for i in range(1, self.no_of_criteria+1):
            names['c' + str(i)] = Dataset.load_from_df(df[['uid','iid','c'+str(i)]], reader = r)
            names['c' + str(i) + '_train'] = names.get('c' + str(i)).build_full_trainset()
            names['algo_c' + str(i)] = SlopeOne()
            names.get('algo_c' + str(i)).fit(names.get('c'+str(i)+'_train'))
            self.algos.append(names.get('algo_c' + str(i)))


    def Predict(self, min_or_more = 'min', pred_options = {'is_total':False, 'combin_func':'avg'}):
        
        names = locals()
        r = Reader(rating_scale=(1,5))
        if min_or_more == 'min':
            df = self.min_test.sort_values(by='uid')
        else:
            df = self.more_test.sort_values(by='uid')
        # df = self.testDatas
        total_test = np.array(df[['uid','iid','total']])
        total_p = self.algos[0].test(total_test)
        for i in range(1, self.no_of_criteria+1):
            # names['c' + str(i) + '_test'] = np.array(df[['uid','iid', 'c' + str(i)]])
            names['c' + str(i) + '_test'] = np.array(df[['uid','iid', 'c' + str(i)]])
            names['c' + str(i) + '_p'] = self.algos[i].test(names.get('c' + str(i) + '_test'))
        
        multi_p = []
        if pred_options['is_total']:
            if pred_options['combin_func'] == 'avg':
                for i in range(len(total_p)):
                    s = 0
                    for j in range(1, self.no_of_criteria+1):
                        s = s + names.get('c'+str(j)+'_p')[i].est
                    avg = (s + total_p[i].est) / (self.no_of_criteria + 1)
                    p = predictions.Prediction(total_p[i].uid, total_p[i].iid, total_p[i].r_ui, avg, total_p[i].details)
                    multi_p.append(p)
            else:
                print('总分作为准则不适合用于回归聚合函数')
        else:
            if pred_options['combin_func'] == 'avg':
                for i in range(len(total_p)):
                    s = 0
                    for j in range(1, self.no_of_criteria+1):
                        s = s + names.get('c'+str(j)+'_p')[i].est
                    avg = s / self.no_of_criteria
                    p = predictions.Prediction(total_p[i].uid, total_p[i].iid, total_p[i].r_ui, avg, total_p[i].details)
                    multi_p.append(p)
            elif pred_options['combin_func'] == 'total_reg':
                k = self.k
                b = self.b
                for i in range(len(total_p)):
                    s = 0
                    for j in range(self.no_of_criteria):
                        s = s + k[j]*names.get('c'+str(j+1)+'_p')[i].est
                    s = s + b
                    p = predictions.Prediction(total_p[i].uid, total_p[i].iid, total_p[i].r_ui, s, total_p[i].details)
                    multi_p.append(p)
            elif pred_options['combin_func'] == 'info_entropy':
                self.Info_Entropy()
                H = np.array(self.H)
                
                for i in range(len(total_p)):
                    s = 0
                    if len(np.argwhere(H[:,0] == total_p[i].uid)):
                        h = H[np.argwhere(H[:,0] == total_p[i].uid)][0][0]
                        for j in range(1, self.no_of_criteria+1):
                            s = s + h[j]*names.get('c'+str(j)+'_p')[i].est/h[self.no_of_criteria+1]
                        
                        p = predictions.Prediction(total_p[i].uid, total_p[i].iid, total_p[i].r_ui, s, total_p[i].details)
                        multi_p.append(p)
                    else:
                        s = 0
                        for j in range(1, self.no_of_criteria+1):
                            s = s + names.get('c'+str(j)+'_p')[i].est
                        s = s/self.no_of_criteria
                        p = predictions.Prediction(total_p[i].uid, total_p[i].iid, total_p[i].r_ui, s, total_p[i].details)
                        multi_p.append(p)


                        
                        
                               
        s_mae = round(accuracy.mae(total_p),4)     
        m_mae = round(accuracy.mae(multi_p),4)        
        return s_mae, m_mae

            
    
    
            
        
            
            
        
        
        
        
        
        
        
            