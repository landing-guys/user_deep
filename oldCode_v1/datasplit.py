import pandas as pd
import numpy as np
import os
from collections import Counter

def split_yahoodata_byk(df,k,datasname):
    #统计每个用户的度（评论数）
    c = Counter(df.uid).most_common()
    data = np.array(c)
    tempid = []
    for i in data:
        if i[1] > k:
            tempid.append(i[0])        
    moredata = df[df['uid'].isin(tempid)]        
    mindata = df[~df['uid'].isin(tempid)]
    path1 = './' + datasname + '/' + str(k) + '/min'
    if not os.path.exists(path1):
        os.makedirs(path1)
    mindata.to_csv(path1 + '/total.txt',columns = ['id', 'uid', 'mid', 'total'], index = False,sep='\t',header = None)
    mindata.to_csv(path1 + '/story.txt',columns = ['id', 'uid', 'mid', 'story'], index = False,sep='\t',header = None)
    mindata.to_csv(path1 + '/role.txt',columns = ['id', 'uid', 'mid', 'role'], index = False,sep='\t',header = None)
    mindata.to_csv(path1 + '/show.txt',columns = ['id', 'uid', 'mid', 'show'], index = False,sep='\t',header = None)
    mindata.to_csv(path1 + '/image.txt',columns = ['id', 'uid', 'mid', 'image'], index = False,sep='\t',header = None)
    mindata.to_csv(path1 + '/music.txt',columns = ['id', 'uid', 'mid', 'music'], index = False,sep='\t',header = None)
    mindata.to_csv(path1 + '/all.txt',columns = ['id', 'uid', 'mid', 'total', 'story', 'role', 'show', 'image', 'music'], index = False,sep='\t',header = None)
    path2 = './' + datasname + '/' + str(k) + '/more'
    if not os.path.exists(path2):
        os.makedirs(path2)
    moredata.to_csv(path2 + '/total.txt',columns = ['id', 'uid', 'mid', 'total'], index = False,sep='\t',header = None)
    moredata.to_csv(path2 + '/story.txt',columns = ['id', 'uid', 'mid', 'story'], index = False,sep='\t',header = None)
    moredata.to_csv(path2 + '/role.txt',columns = ['id', 'uid', 'mid', 'role'], index = False,sep='\t',header = None)
    moredata.to_csv(path2 + '/show.txt',columns = ['id', 'uid', 'mid', 'show'], index = False,sep='\t',header = None)
    moredata.to_csv(path2 + '/image.txt',columns = ['id', 'uid', 'mid', 'image'], index = False,sep='\t',header = None)
    moredata.to_csv(path2 + '/music.txt',columns = ['id', 'uid', 'mid', 'music'], index = False,sep='\t',header = None)    
    moredata.to_csv(path2 + '/all.txt',columns = ['id', 'uid', 'mid', 'total', 'story', 'role', 'show', 'image', 'music'], index = False,sep='\t',header = None)    


def split_tripadvisor_byk(df,k,datasname):
    #统计每个用户的度（评论数）
    c = Counter(df.uid).most_common()
    data = np.array(c)
    tempid = []
    for i in data:
        if i[1] > k:
            tempid.append(i[0])        
    moredata = df[df['uid'].isin(tempid)]        
    mindata = df[~df['uid'].isin(tempid)]
    path1 = './' + datasname + '/' + str(k) + '/min'
    if not os.path.exists(path1):
        os.makedirs(path1)
    mindata.to_csv(path1 + '/total.txt',columns = ['id', 'uid', 'hid', 'total'], index = False,sep='\t',header = None)
    mindata.to_csv(path1 + '/xinJiaBi.txt',columns = ['id', 'uid', 'hid', 'xinJiaBi'], index = False,sep='\t',header = None)
    mindata.to_csv(path1 + '/shuShiDu.txt',columns = ['id', 'uid', 'hid', 'shuShiDu'], index = False,sep='\t',header = None)
    mindata.to_csv(path1 + '/weiZi.txt',columns = ['id', 'uid', 'hid', 'weiZi'], index = False,sep='\t',header = None)
    mindata.to_csv(path1 + '/weiShen.txt',columns = ['id', 'uid', 'hid', 'weiShen'], index = False,sep='\t',header = None)
    # mindata.to_csv(path1 + '/shuiMian.txt',columns = ['id', 'uid', 'hid', 'shuiMian'], index = False,sep='\t',header = None)    
    mindata.to_csv(path1 + '/fuWu.txt',columns = ['id', 'uid', 'hid', 'fuWu'], index = False,sep='\t',header = None)       
    mindata.to_csv(path1 + '/all.txt',columns = ['id', 'uid', 'hid', 'total', 'xinJiaBi', 'shuShiDu', 'weiZi', 'weiShen', 'fuWu'], index = False,sep='\t',header = None)    
    path2 = './' + datasname + '/' + str(k) + '/more'
    if not os.path.exists(path2):
        os.makedirs(path2)
    moredata.to_csv(path2 + '/total.txt',columns = ['id', 'uid', 'hid', 'total'], index = False,sep='\t',header = None)
    moredata.to_csv(path2 + '/xinJiaBi.txt',columns = ['id', 'uid', 'hid', 'xinJiaBi'], index = False,sep='\t',header = None)
    moredata.to_csv(path2 + '/shuShiDu.txt',columns = ['id', 'uid', 'hid', 'shuShiDu'], index = False,sep='\t',header = None)
    moredata.to_csv(path2 + '/weiZi.txt',columns = ['id', 'uid', 'hid', 'weiZi'], index = False,sep='\t',header = None)
    moredata.to_csv(path2 + '/weiShen.txt',columns = ['id', 'uid', 'hid', 'weiShen'], index = False,sep='\t',header = None)
    # moredata.to_csv(path2 + '/shuiMian.txt',columns = ['id', 'uid', 'hid', 'shuiMian'], index = False,sep='\t',header = None)    
    moredata.to_csv(path2 + '/fuWu.txt',columns = ['id', 'uid', 'hid', 'fuWu'], index = False,sep='\t',header = None)       
    moredata.to_csv(path2 + '/all.txt',columns = ['id', 'uid', 'hid', 'total', 'xinJiaBi', 'shuShiDu', 'weiZi', 'weiShen', 'fuWu'], index = False,sep='\t',header = None)    

# 读取文件为dataframe类型
datasname = 'tripadvisor'
df = pd.read_csv('./Datas/'+ datasname + '.txt', sep = '\t', names = ['id', 'uid', 'hid', 'total', 'xinJiaBi', 'shuShiDu', 'weiZi', 'weiShen', 'shuiMian', 'fuWu'])
#以k为阈值划分数据集
for k in range(1,21,1):
    # split_yahoodata_byk(df,k,datasname)
    split_tripadvisor_byk(df, k, datasname)



    

