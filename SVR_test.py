# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 21:04:24 2020

@author: Jerry
"""



import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


read_path = './Datas/alldata-188854.txt'
colnames = ['id','uid','iid','total'] + ['c'+str(i+1) for i in range(5)]
df = pd.read_csv(read_path, sep = '\t', names = colnames)

df = df[df['uid'].isin([8])]

X = df[['c'+str(i+1) for i in range(5)]].values
y = df[['total']].values

X = np.reshape(X, (-1, 5))
y = np.reshape(y, (-1, 1))


# sc_X = StandardScaler()
# sc_y = StandardScaler()
# X = sc_X.fit_transform(X)
# y = sc_y.fit_transform(y)

# y = [t[0] for t in y]
kernels = ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed']
svr = SVR()
svr.fit(X, y)
p = svr.predict(X)
# score_svr = round(r2_score(y, p),3)
# lin = LinearRegression()
# lin.fit(X, y)
# p2 = lin.predict(X)
# score_lin = round(r2_score(y, p2),3)
# print('SVR Accruacy is {0}\nLin Accruacy is {1}'.format(score_svr, score_lin))
# x = np.linalg.norm(X,ord=None,axis=1)
# x = [np.e**-(t**2) for t in x]
# for t in y:
#     x.append(t)
# for t in p:
#     x.append(t)
# for t in p2:
#     x.append(t)

# x = np.reshape(x, (-1,4), order = 'F')
# x = x[x[:,0].argsort()]#按第一列排序
# plt.scatter(x[:,0], x[:,1], color = 'red')
# plt.plot(x[:,0], x[:,2], color = 'blue')
# plt.plot(x[:,0], x[:,3], color = 'green')
# plt.title('Truth or Bluff (SVR)')
# plt.xlabel('Position level')
# plt.ylabel('Salary')
# plt.show()


# x = []

# for i in range(1, 100001, 1):
#     svr = SVR(kernel = 'rbf',C = i/100)
#     svr.fit(X, y)
#     p = svr.predict(X)
#     score_svr = round(r2_score(y, p),3)
#     x.append((i/100,score_svr))

# x = np.array(x)
# plt.plot(x[:,0], x[:,1], color = 'blue')
# plt.ylim(min(x[:,1]),max(x[:,1]))
# plt.title('Truth or Bluff (SVR)')
# plt.xlabel('gamma')
# plt.ylabel('Accuracy')
# plt.show()

# C_best = x[x[:,1].argsort()][-1][0]
# C_s = x[x[:,1].argsort()][-1][1]
# print('The best C is:{0}\nAnd its Accruacy is {1}'.format(C_best, C_s))

# x2 = []

# for i in range(1, 151, 2):
#     svr = SVR(kernel = 'rbf', gamma=i/100)
#     svr.fit(X, y)
#     p = svr.predict(X)
#     score_svr = round(r2_score(y, p),3)
#     x2.append((i/100,score_svr))

# plt.figure()
# x2 = np.array(x2)
# plt.plot(x2[:,0], x2[:,1], color = 'blue')
# plt.title('Truth or Bluff (SVR)')
# plt.xlabel('gamma')
# plt.ylabel('Accuracy')
# plt.show()

# gamma_best = x2[x2[:,1].argsort()][-1][0]  
# gamma_s = x2[x2[:,1].argsort()][-1][1]  
# print('The best gamma is:{0}\nAnd its Accruacy is {1}'.format(gamma_best, gamma_s))

    



















