import pandas as pd
import numpy as np
from math import ceil
from math import floor
import os
from collections import Counter

class EDatas:#Experimental Datas
    ''' path：str,初始数据的存储路径
        no_of_cirteria：int,准则的数量
        steps：int,划分数据的步长
        ways：str,划分的方式， ‘interval’以区间的形式划分；‘step’逐步划分数据
        maxthreshold:int,划分数据的最大阈值
        datas：DataFrame,存放数据'''
    # no_of_criteria = 5
    # steps = 1
    # ways = 'step'
    # maxthreshold=20
    
    def __init__(self, no_of_criteria=5, steps=1, ways='step', maxthreshold=20, datas = []):
        # self.path = path
        self.no_of_criteria = no_of_criteria
        self.steps = steps
        self.ways = ways
        self.maxthreshold = maxthreshold
        self.datas = datas

        
    
    def readDatas(self, inpath):
        colnames = ['id','uid','iid','total'] + ['c'+str(i+1) for i in range(self.no_of_criteria)]
        self.datas = pd.read_csv(inpath, sep = '\t', names = colnames)
        
    def splitDatas(self, outpath, random_state = 666):
        '''outpath:存储划分后数据的文件夹的名字'''
        outpath2 = outpath + '/test'
        df = self.datas
        trainset, testset = self.train_test_split(len(df), random_state)
        train = df[df['id'].isin(trainset)]
        if not os.path.exists(outpath):
            os.makedirs(outpath)
        train.to_csv(outpath + '/train.txt',columns = ['id','uid','iid','total'] + ['c'+str(x+1) for x in range(self.no_of_criteria)], index = False,sep='\t',header = None)
        test = df[df['uid'].isin(testset)]
        c = Counter(test.uid).most_common()
        data = np.array(c)
        if self.ways == 'step':
            for k in range(1, self.maxthreshold+1, self.steps):
                tempid = []
                for i in data:
                    if i[1] > k:
                        tempid.append(i[0])        
                moredata = test[test['uid'].isin(tempid)]        
                mindata = test[~test['uid'].isin(tempid)]
                pathmin = './' + outpath2 + '/' + str(k) + '/min'
                if not os.path.exists(pathmin):
                    os.makedirs(pathmin)
                pathmax = './' + outpath2 + '/' + str(k) + '/more'
                if not os.path.exists(pathmax):
                    os.makedirs(pathmax)
                # for j in range(self.no_of_criteria):
                #     mindata.to_csv(pathmin+'/c'+str(j+1)+'.txt',columns=['id','uid','iid','c'+str(j+1)],index = False, sep='\t', header = None)
                #     moredata.to_csv(pathmax+'/c'+str(j+1)+'.txt',columns=['id','uid','iid','c'+str(j+1)],index = False, sep='\t', header = None)
                # mindata.to_csv(pathmin + '/total.txt',columns = ['id', 'uid', 'iid', 'total'], index = False,sep='\t',header = None)
                mindata.to_csv(pathmin + '/all.txt',columns = ['id','uid','iid','total'] + ['c'+str(x+1) for x in range(self.no_of_criteria)], index = False,sep='\t',header = None)
                # moredata.to_csv(pathmax + '/total.txt',columns = ['id', 'uid', 'iid', 'total'], index = False,sep='\t',header = None)
                moredata.to_csv(pathmax + '/all.txt',columns = ['id','uid','iid','total'] + ['c'+str(x+1) for x in range(self.no_of_criteria)], index = False,sep='\t',header = None) 
        else:
            for k in range(0,self.maxthreshold+1, self.steps):
                tempid = []
                if k+self.steps <= self.maxthreshold:   
                    for i in data:
                        if i[1] > k and i[1] <= k+self.steps:
                            tempid.append(i[0])
                    tempdata = test[test['uid'].isin(tempid)]
                    temppath = './' + outpath2 + '/' + str(k+1) + '-'  + str(k+self.steps) 
                    if not os.path.exists(temppath):
                        os.makedirs(temppath)
                    # for j in range(self.no_of_criteria):
                    #     tempdata.to_csv(temppath+'/c'+str(j+1)+'.txt',columns=['id','uid','iid','c'+str(j+1)],index = False, sep='\t', header = None)
                    # tempdata.to_csv(temppath + '/total.txt',columns = ['id', 'uid', 'iid', 'total'], index = False,sep='\t',header = None)
                    tempdata.to_csv(temppath + '/all.txt',columns = ['id','uid','iid','total'] + ['c'+str(x+1) for x in range(self.no_of_criteria)], index = False,sep='\t',header = None)
                else:
                    for i in data:
                        if i[1] > k:
                            tempid.append(i[0])
                    tempdata = test[test['uid'].isin(tempid)]
                    temppath = './' + outpath2 + '/' + str(k+1) + '-all'
                    if not os.path.exists(temppath):
                        os.makedirs(temppath)
                    # for j in range(self.no_of_criteria):
                    #     tempdata.to_csv(temppath+'/c'+str(j+1)+'.txt',columns=['id','uid','iid','c'+str(j+1)],index = False, sep='\t', header = None)
                    # tempdata.to_csv(temppath + '/total.txt',columns = ['id', 'uid', 'iid', 'total'], index = False,sep='\t',header = None)
                    tempdata.to_csv(temppath + '/all.txt',columns = ['id','uid','iid','total'] + ['c'+str(x+1) for x in range(self.no_of_criteria)], index = False,sep='\t',header = None)

    def train_test_split(self, n_ratings, random_state):
        test_size, train_size = self.validate_train_test_sizes(n_ratings)
        rng = np.random.RandomState(random_state)
        permutation = rng.permutation(n_ratings)+1
        trainset = permutation[:test_size]
        testset = permutation[test_size:(test_size + train_size)]
        return trainset, testset
    
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
    



    




    

