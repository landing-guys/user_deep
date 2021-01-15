# -*- coding: utf-8 -*-
"""
Created on Sat Oct 10 16:34:40 2020

@author: Jerry
"""
import os
from RecSys import Experiment
import matplotlib.pyplot as plt

def plot_mae(x, maes, filepath, strs):
    plt.rcParams['axes.unicode_minus']=False
    plt.rcParams['font.sans-serif']=['SimHei']
    y1,y2 = [], []
    for m in maes:
        y1.append(m[0])
        y2.append(m[1])
    fig = plt.figure()
    plt.grid(axis="y")
    plt.plot(x, y1, label='单准则')
    plt.plot(x, y2, label='多准则')
    plt.xlabel('用户度范围')
    plt.ylabel('mae')
    plt.title(strs)
    fig.legend()
    fig.savefig(filepath)
    fig.show() 

def plot_mae_increase(x, maes, path):
    plt.rcParams['axes.unicode_minus']=False
    plt.rcParams['font.sans-serif']=['SimHei']
    y1,y2 = [], []
    for m in maes:
        y1.append(m[0])
        y2.append(m[1])
    fig = plt.figure()
    plt.grid(axis="y")
    plt.plot(x, y1, label='小度')
    plt.plot(x, y2, label='大度')
    plt.xlabel('用户度划分阈值')
    plt.ylabel('mae提升率')
    plt.title('不同用户度阈值下大度用户小度用户的mae提升率')
    fig.legend()
    fig.savefig(path)
    fig.show()

N = 20

indirs = './Datas/'

split_option = {'k_item':10,'k_user':5,'random_state':666}
outdirs = './item-10-user-5'

# KNN's fit_options
# fit_options = {'name': 'pearson', 'user_based': False}
# SVD's fit_options
fit_options = {'n_factors': 50, 'n_epochs': 30, 'random_state': 0}
# KNN's filenames
filenames = outdirs + '/SVD-notttotal-info_entropy/'
# SVD's filenames
# filenames = outdirs + '/SVD-50-25-nottotal-total_reg/'
pred_option = {'is_total':False, 'combin_func':'info_entropy'}

if not os.path.exists(filenames):
    os.makedirs(filenames)
min_maes, max_maes = [], []
E = Experiment(no_of_criteria = 5)
#训练
E.read_split_Datas(indirs + 'yahoo.txt', split_options=split_option)
# E.KNN_train(options = fit_options)
E.SVD_train(options = fit_options)
# E.SlopeOne_train()
# 预测
for i in range(N):
    #小度
    E.splitTestDatas(i+1)
    mae1, mae2 = E.Predict('min', pred_options = pred_option)
    min_maes.append((mae1, mae2))
    #大度
    mae1, mae2 = E.Predict('more', pred_options = pred_option)#combin_func = 'total_reg'
    max_maes.append((mae1, mae2))
    


mae_increase = []
for i in range(len(min_maes)):
    min_increase = round(100*(min_maes[i][0]-min_maes[i][1]) / min_maes[i][0],4)
    max_increase = round(100*(max_maes[i][0]-max_maes[i][1]) / max_maes[i][0],4)
    mae_increase.append((min_increase, max_increase))
names = [filenames  + 'mae具体值.txt', filenames + 'mae提升率.txt', filenames + 'mae提升率对比图.png', filenames + '小度用户mae对比图.png', filenames + '大度用户mae对比图.png']

for n in names:
    if(os.path.exists(n)):
        os.remove(n)
for i in range(len(min_maes)):
    f1 = open(names[0],'a+')
    f1.write(str(i+1) + '\t' + str(min_maes[i][0]) + '\t' + str(min_maes[i][1]) + '\t' + str(max_maes[i][0])  + '\t' + str(max_maes[i][1]) +'\n')
    f1.close()
    f2 = open(names[1],'a+')
    f2.write(str(i+1) + '\t' + str(mae_increase[i][0]) + '\t' + str(mae_increase[i][1]) + '\n')
    f2.close()
plot_mae_increase([str(i+1) for i in range(N)], mae_increase, names[2])
plot_mae([str(i+1) for i in range(N)], min_maes, names[3],'小度')
plot_mae([str(i+1) for i in range(N)], max_maes, names[4],'大度')