# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 19:41:20 2020

@author: Jerry
"""

import time
time_start = time.time()
import pandas as pd
import numpy as np
from math import log2
from collections import Counter

colnames = ['id','uid','iid','total'] + ['c'+str(i+1) for i in range(5)]
df = pd.read_csv('./Datas/yahoo.txt', sep = '\t', names = colnames)
df = df.sort_values(by='uid')
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
            th = - sum([j*log2(j) for j in p])
            total_h = total_h + th
            h.append(th)
        h.append(total_h)
        if total_h == 0:
            h = [uid,1,1,1,1,1,5]
        
    else:
        for i in range(5):
            h.append(1)
        h.append(5)
    H.append(h)
time_end = time.time()
print('totally cost{0}s'.format(time_end-time_start))