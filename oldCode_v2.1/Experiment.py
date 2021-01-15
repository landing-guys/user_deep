# -*- coding: utf-8 -*-
"""
Created on Sat Sep 26 21:41:29 2020

@author: Jerry
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from surprise import Dataset
from surprise import Reader
from surprise import KNNWithMeans
from surprise import accuracy
from surprise import SVD
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
    
    
    def __init__(self, no_of_criteria=5, traindatas = [], testdatas = [], algos = []):
        # self.combin_func = combin_func
        # self.is_total = is_total
        self.no_of_criteria = no_of_criteria
        self.traindatas = traindatas
        self.testdatas = testdatas
        self.algos = algos

    
    def total_reg_param(self):
        x = self.traindatas[['c' + str(i) for i in range(1, self.no_of_criteria+1)]]
        y = self.traindatas[['total']]
        x = np.array(x)
        y = np.array(y)                
        reg = LinearRegression()
        reg.fit(x, y)
        self.k = reg.coef_[0]
        self.b = reg.intercept_[0]
        
    def readTrainDatas(self, read_path):
        colnames = ['id','uid','iid','total'] + ['c'+str(i+1) for i in range(self.no_of_criteria)]
        self.traindatas = pd.read_csv(read_path, sep = '\t', names = colnames)
        self.total_reg_param()
    
    def readTestDatas(self, read_path):
        self.testdatas = []
        colnames = ['id','uid','iid','total'] + ['c'+str(i+1) for i in range(self.no_of_criteria)]
        self.testdatas = pd.read_csv(read_path, sep = '\t', names = colnames)


        
    def KNN_train(self, k = 20, options = {'name': 'pearson', 'user_based': False}):
        '''
        seed：int-3划分训练集测试集的随机种子
        k：int-40，最大邻居数量
        options：dict-{'name': 'pearson', 'user_based': False}，算法的选项，默认为Pearson相似度，基于项目的方法
        '''
        self.algos = []
        df = self.traindatas
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
        


    def KNN_pred(self, is_total = 0, combin_func = 'avg'):
        names = locals()
        r = Reader(rating_scale=(1,5))
        df = self.testdatas
        total_test = np.array(df[['uid','iid','total']])
        total_p = self.algos[0].test(total_test)
        for i in range(1, self.no_of_criteria+1):
            # names['c' + str(i) + '_test'] = np.array(df[['uid','iid', 'c' + str(i)]])
            names['c' + str(i) + '_test'] = Dataset.load_from_df(df[['uid','iid', 'c' + str(i)]], reader = r)
            names['c' + str(i) + '_test'] = names.get('c' + str(i) + '_test').build_full_trainset()
            names['c' + str(i) + '_test'] = names.get('c' + str(i) + '_test').build_
            names['c' + str(i) + '_p'] = self.algos[i].test(names.get('c' + str(i) + '_test'))
        
        multi_p = []
        if is_total == 0:
            if combin_func == 'avg':
                for i in range(len(total_p)):
                    s = 0
                    for j in range(1, self.no_of_criteria+1):
                        s = s + names.get('c'+str(j)+'_p')[i].est
                    avg = s / self.no_of_criteria
                    p = predictions.Prediction(total_p[i].uid, total_p[i].iid, total_p[i].r_ui, avg, total_p[i].details)
                    multi_p.append(p)
            elif combin_func == 'total_reg':
                k = self.k
                b = self.b
                for i in range(len(total_p)):
                    s = 0
                    for j in range(self.no_of_criteria):
                        s = s + k[j]*names.get('c'+str(j+1)+'_p')[i].est
                    s = s + b
                    p = predictions.Prediction(total_p[i].uid, total_p[i].iid, total_p[i].r_ui, s, total_p[i].details)
                    multi_p.append(p)
        else:
            if combin_func == 'avg':
                for i in range(len(total_p)):
                    s = 0
                    for j in range(1, self.no_of_criteria+1):
                        s = s + names.get('c'+str(j)+'_p')[i].est
                    avg = (s + total_p[i].est) / (self.no_of_criteria + 1)
                    p = predictions.Prediction(total_p[i].uid, total_p[i].iid, total_p[i].r_ui, avg, total_p[i].details)
                    multi_p.append(p)
            else:
                print('总分作为准则不适合用于回归聚合函数')
        s_mae = round(accuracy.mae(total_p),4)     
        m_mae = round(accuracy.mae(multi_p),4)        
        return s_mae, m_mae, total_p, multi_p
    
    def SVD(self, seed = 3, n_factor = 20, n_epoch = 20):
        '''
        seed：int-3划分训练集测试集的随机种子
        k：int-40，最大邻居数量
        options：dict-{'name': 'pearson', 'user_based': False}，算法的选项，默认为Pearson相似度，基于项目的方法
        '''
        df = self.datas
        names = locals()
        r = Reader(rating_scale=(1,5))
        # 读取、划分数据;训练预测数据
        total = Dataset.load_from_df(df[['uid','iid','total']], reader = r)
        total_train, total_test = train_test_split(total, random_state = seed)
        total_algo = SVD(n_factors = n_factor, n_epochs = n_epoch,verbose = True)
        total_algo.fit(total_train)
        total_p = total_algo.test(total_test)
        for i in range(1, self.no_of_criteria+1):
            names['c' + str(i)] = Dataset.load_from_df(df[['uid','iid','c'+str(i)]], reader = r)
            names['c' + str(i) + '_train'], names['c' + str(i) + '_test'] = train_test_split(names.get('c' + str(i)), random_state = seed)
            names['algo_c' + str(i)] = SVD(n_factors = n_factor, n_epochs = n_epoch, verbose = True)
            names.get('algo_c' + str(i)).fit(names.get('c'+str(i)+'_train'))
            names['c' + str(i) + '_p'] = names.get('algo_c' + str(i)).test(names.get('c'+str(i)+'_test'))
        #
        multi_p = []
        if self.is_total == 0:
            if self.combin_func == 'avg':
                for i in range(len(total_p)):
                    s = 0
                    for j in range(1, self.no_of_criteria+1):
                        s = s + names.get('c'+str(j)+'_p')[i].est
                    avg = s / self.no_of_criteria
                    p = predictions.Prediction(total_p[i].uid, total_p[i].iid, total_p[i].r_ui, avg, total_p[i].details)
                    multi_p.append(p)
            elif self.combin_func == 'total_reg':
                for i in range(1, self.no_of_criteria+1):
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
                    for j in range(self.no_of_criteria):
                        s = s + k[j]*names.get('c'+str(j+1)+'_p')[i].est
                    s = s +b
                    p = predictions.Prediction(total_p[i].uid, total_p[i].iid, total_p[i].r_ui, s, total_p[i].details)
                    multi_p.append(p)
        else:
            if self.combin_func == 'avg':
                for i in range(len(total_p)):
                    s = 0
                    for j in range(1, self.no_of_criteria+1):
                        s = s + names.get('c'+str(j)+'_p')[i].est
                    avg = (s + total_p[i].est) / (self.no_of_criteria + 1)
                    p = predictions.Prediction(total_p[i].uid, total_p[i].iid, total_p[i].r_ui, avg, total_p[i].details)
                    multi_p.append(p)
            else:
                print('总分作为准则不适合用于回归聚合函数')
        s_mae = round(accuracy.mae(total_p),4)     
        m_mae = round(accuracy.mae(multi_p),4)        
        return s_mae, m_mae
    
    
            
        
            
            
        
        
        
        
        
        
        
            