import pandas as pd #导入pandas库
import numpy as np #导入numpy库
import os
from collections import Counter
from sklearn.linear_model import LinearRegression

#将电影评分少于k的数据剔除
def filterDatasbyMid(df, k):
    c = Counter(df.mid).most_common()
    tempid = []
    data = np.array(c)
    for d in data:
        if d[1] < k:
            tempid.append(d[0])
    dfs = df[~df['mid'].isin(tempid)]
    return dfs


def totalRegModel(df):
    data_train = np.array(df[['story','role','show','image','music']]).reshape(len(df),5)
    data_test = np.array(df['total']).reshape(len(df),1) 
    #整个数据集的回归模型
    reg = LinearRegression()
    reg.fit(data_train,data_test)
    k = reg.coef_[0]
    b = reg.intercept_[0]
    return k, b


def userRegModel(df):
    c = Counter(df.uid).most_common()
    uid = []
    data = np.array(c)
    param = []
    k, b = totalRegModel(df)
    param.append((0,k,b))
    for d in data:
        uid.append([d[0]])
    for u in uid:
        dfs = df[df['uid'].isin(u)]
        data_train = np.array(dfs[['story','role','show','image','music']]).reshape(len(dfs),5)
        data_test = np.array(dfs['total']).reshape(len(dfs),1)
        reg = LinearRegression()
        reg.fit(data_train,data_test)
        k = reg.coef_[0]
        b = reg.intercept_[0]
        param.append((u[0],k,b))
    return param


datasname = 'alldata-188854'
df = pd.read_csv('./Datas/'+ datasname + '.txt', sep = '\t', names = ['id', 'uid', 'mid', 'total', 'story', 'role', 'show', 'image', 'music'])
k,b = totalRegModel(df)
param = userRegModel(df)
    


#每个用户的回归模型


#先按照用户id划分数据


#给每个用户建立回归模型并保存
