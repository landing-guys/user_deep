# -*- coding: utf-8 -*-
"""
Created on Sat Aug 15 16:43:01 2020

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
from surprise.prediction_algorithms import predictions

from sklearn.linear_model import LinearRegression





def chose_yahoo(train_path, test_path, k):
    train_df = pd.read_csv(train_path, sep = '\t', names = ['id', 'uid', 'mid', 'total', 'story', 'role', 'show', 'image', 'music'])
    test_df =  pd.read_csv(test_path, sep = '\t', names = ['id', 'uid', 'mid', 'total', 'story', 'role', 'show', 'image', 'music'])
    r = Reader(rating_scale=(1,5))
    story = Dataset.load_from_df(train_df[['uid','mid','story']],reader = r)
    story_train = story.build_full_trainset()
    story_test = np.array(test_df[['uid','mid','story']])
    
    role = Dataset.load_from_df(train_df[['uid','mid','role']],reader = r)
    role_train = role.build_full_trainset()
    role_test = np.array(test_df[['uid','mid','role']])
    
    show = Dataset.load_from_df(train_df[['uid','mid','show']],reader = r)
    show_train = show.build_full_trainset()
    show_test = np.array(test_df[['uid','mid','show']])
    
    image = Dataset.load_from_df(train_df[['uid','mid','image']],reader = r)
    image_train = image.build_full_trainset()
    image_test = np.array(test_df[['uid','mid','image']])
    
    music = Dataset.load_from_df(train_df[['uid','mid','music']],reader = r)
    music_train = music.build_full_trainset()
    music_test = np.array(test_df[['uid','mid','music']])
    
    total = Dataset.load_from_df(train_df[['uid','mid','total']],reader = r)
    total_train = total.build_full_trainset()
    total_test = np.array(test_df[['uid','mid','total']])
    
    sim_options = {'name': 'pearson', 'user_based': False}#用皮尔森基线相似度避免出现过拟合
    # algo1 = KNNWithMeans(k,sim_options=sim_options)
    # algo2 = KNNWithMeans(k,sim_options=sim_options)
    # algo3 = KNNWithMeans(k,sim_options=sim_options)
    # algo4 = KNNWithMeans(k,sim_options=sim_options)
    # algo5 = KNNWithMeans(k,sim_options=sim_options)
    # algo6 = KNNWithMeans(k,sim_options=sim_options)
    
    algo1 = SVD()
    algo2 = SVD()
    algo3 = SVD()
    algo4 = SVD()
    algo5 = SVD()
    algo6 = SVD()
    
    algo1.fit(story_train)
    story_p = algo1.test(story_test) 
    
    algo2.fit(role_train)
    role_p = algo2.test(role_test)
    
    algo3.fit(show_train)
    show_p = algo3.test(show_test)
    
    algo4.fit(image_train)
    image_p = algo4.test(image_test)
    
    algo5.fit(music_train)
    music_p = algo5.test(music_test)
    
    algo6.fit(total_train)
    total_p = algo6.test(total_test)
    
    P = combine(story_p, role_p, show_p, image_p, music_p, total_p)
    print('数据合并完毕')
    #平均模型
    multi_p = avg(P, total_p)
    #整体回归模型
    # k, b = get_param_from_totaldata(train_df)
    # multi_p = totaldata_regModel(P, k, b, total_p)
    #每个用户的回归模型
    print('多准则评分计算完毕')
    mae = (round(accuracy.mae(total_p),4),round(accuracy.mae(multi_p),4))
    
    return mae

def get_param_from_totaldata(train_df):
    #整体评分不做准则
    data_train = np.array(train_df.iloc[:,4:9]).reshape(len(train_df),5)
    data_test = np.array(train_df.iloc[:,3]).reshape(len(train_df),1)
    reg = LinearRegression()
    reg.fit(data_train, data_test)
    k = reg.coef_[0]
    b = reg.intercept_[0]
    return k, b

def totaldata_regModel(P, k, b, total_p):
    multi_p = []
    for i in range(len(P)):
        #整体评分不做准则
        pred = P[i][3]*k[0] + P[i][4]*k[1] + P[i][5]*k[2] + P[i][6]*k[3] + P[i][7]*k[4] + b
        p = predictions.Prediction(P[i][0], P[i][1], total_p[i][2], pred, total_p[i][4])
        multi_p.append(p)
    return multi_p

def avg(P, total_p):
    multi_p = []
    for i in range(len(P)):
        pred = (P[i][3] + P[i][4] + P[i][5] + P[i][6] + P[i][7] + P[i][2])/6#整体评分作准则
        # pred = (P[i][3] + P[i][4] + P[i][5] + P[i][6] + P[i][7] )/5  #整体评分不做准则
        p = predictions.Prediction(P[i][0], P[i][1], total_p[i][2], pred, total_p[i][4])
        multi_p.append(p)
    return multi_p

def combine(p1, p2, p3, p4, p5, p6):
    P = []
    for i in range(len(p1)):
        P.append((p6[i].uid, p6[i].iid, p6[i].est, p1[i].est,p2[i].est,p3[i].est,p4[i].est,p5[i].est))
    return P


def shows(mae, filepath):
    plt.rcParams['axes.unicode_minus']=False
    plt.rcParams['font.sans-serif']=['SimHei']
    x,y1,y2 = [], [], []
    for m in mae:
        x.append(m[0])
        y1.append(m[1])
        y2.append(m[2])
    fig = plt.figure()
    plt.grid(axis="y")
    plt.plot(x, y1, label='小度')
    plt.plot(x, y2, label='大度')
    plt.xlabel('阈值')
    plt.ylabel('算法提升率(%)')
    plt.title('不同阈值下多准则算法相较于单准则算法对不同度用户的精度提升率')
    fig.legend()
    fig.savefig(filepath)
    fig.show()

def main(k, datasname):
    #datasname = 'yahoodata'
    d_mae, x_mae = [],[]
    # d_rmse, x_rmse = [],[]
    file_path = './' + datasname + '/' + str(k) # './yahoodata/k'
    # file_path = './' + str(k) 
    x_mae = chose_yahoo(file_path + '/min/train.txt', file_path + '/min/test.txt', 1)
    d_mae = chose_yahoo(file_path + '/more/train.txt', file_path + '/more/test.txt', 40)
    return x_mae, d_mae # d_rmse, x_rmse    
    
mae = []
maes = []
datasname = 'tripadvisor(v1)'
for k in range(5,21,1):
    x_mae,d_mae = main(k,datasname)
    t1 = (x_mae[0]-x_mae[1])/x_mae[0] 
    t2 = (d_mae[0]-d_mae[1])/d_mae[0]
    print('阈值为{0}时小度数据下多准则算法精度提升率为{1}%'.format(k,-t1*100))
    print('阈值为{0}时大度数据下多准则算法精度提升率为{1}%'.format(k,-t2*100))
    mae.append((k, round(-t1*100,2), round(-t2*100,2)))
    maes.append((k, x_mae[0], x_mae[1], d_mae[0], d_mae[1]))
    
f = './(改)实验结果/平均（整体评分作准则）SVD/' + datasname 

if not os.path.exists(f):
        os.makedirs(f)
names = ['/mae具体值.txt','/mae提升率.txt','/mae提升率对比图.png']

fs = []
for n in names:
    fs.append(f+n)
for f in fs:
    if(os.path.exists(f)):
        os.remove(f)
for m in maes:
    f1 = open(fs[0],'a+')
    f1.write(str(m[0]) + '\t' + str(m[1]) + '\t' + str(m[2])  + '\t' + str(m[3])  + '\t' + str(m[4]) +'\n')
    f1.close()
for m in mae:
    f2 = open(fs[1],'a+')
    f2.write(str(m[0]) + '\t' + str(m[1]) + '\t' + str(m[2]) +'\n')
    f2.close()
shows(mae, fs[2])