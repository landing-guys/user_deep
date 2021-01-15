import pandas as pd
import numpy as np
from surprise import SVD
from surprise import Dataset
from surprise import Reader
from surprise.model_selection import cross_validate


# Load the movielens-100k dataset (download it if needed),

path = './yahoodata(step)/6/min/all.txt'
colnames = ['id','uid','iid','total'] + ['c'+str(i+1) for i in range(5)]
df = pd.read_csv(path, sep = '\t', names = colnames)
r = Reader(rating_scale=(1,5))
# 读取、划分数据;训练预测数据
data = Dataset.load_from_df(df[['uid','iid','total']], reader = r)

# n1 = 50
# n2 = 25
'''1 min: n1 = 50, n2 = 30
  1 more: n1 = 50, n2 = 25
   2 min: n1 = 100,n2 = 30
  2 more: n1 = 50, n2 = 30
   3 min: n1 = 100,n2 = 30
  3 more: n1 = 50, n2 = 30
   4 min: n1 = 100,n2 = 30
  4 more: n1 = 50, n2 = 25
   5 min: n1 = 50, n2 = 30
  5 more: n1 = 50, n2 = 25
   6 min: n1 = 50,n2 = 30
  6 more: n1 = 50, n2 = 30
   '''
temp_mae = 2
n_factor = 0
n_epoch = 0
for n1 in range(50,200,50):
    for n2 in range(20,35,5):
        algo = SVD(n_factors = n1, n_epochs = n2)
        k = cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)
        mae = np.mean(k['test_mae'])
        if mae < temp_mae:
            temp_mae = mae
            n_factor = n1
            n_epoch = n2