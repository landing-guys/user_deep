# -*- coding: utf-8 -*-
"""
Created on Fri Aug 14 20:46:19 2020

@author: Jerry
"""
import pandas as pd
import os 

def split(k,in_file):
    min_in_file = in_file + str(k) + '/min/'
    more_in_file = in_file + str(k) + '/more/'
    splitdata(min_in_file)
    splitdata(more_in_file)
    
def splitdata(file):
    train_in_file = file + 'train.txt'
    test_in_file = file + 'test.txt'
    train = pd.read_csv(train_in_file, sep = '\t', names = ['id', 'uid', 'mid', 'total', 'story', 'role', 'show', 'image', 'music'])
    test = pd.read_csv(test_in_file, sep = '\t', names = ['id', 'uid', 'mid', 'total', 'story', 'role', 'show', 'image', 'music'])
    train_out_path = file + '/train/'
    test_out_path = file + '/test/'
    if not os.path.exists(train_out_path):
        os.makedirs(train_out_path)
    train.to_csv(train_out_path + '/total.txt',columns = ['id', 'uid', 'mid', 'total'], index = False,sep='\t',header = None)
    train.to_csv(train_out_path + '/story.txt',columns = ['id', 'uid', 'mid', 'story'], index = False,sep='\t',header = None)
    train.to_csv(train_out_path + '/role.txt',columns = ['id', 'uid', 'mid', 'role'], index = False,sep='\t',header = None)
    train.to_csv(train_out_path + '/show.txt',columns = ['id', 'uid', 'mid', 'show'], index = False,sep='\t',header = None)
    train.to_csv(train_out_path + '/image.txt',columns = ['id', 'uid', 'mid', 'image'], index = False,sep='\t',header = None)
    train.to_csv(train_out_path + '/music.txt',columns = ['id', 'uid', 'mid', 'music'], index = False,sep='\t',header = None)    
    train.to_csv(train_out_path + '/all.txt',columns = ['id', 'uid', 'mid', 'total', 'story', 'role', 'show', 'image', 'music'], index = False,sep='\t',header = None)    
    if not os.path.exists(test_out_path):
        os.makedirs(test_out_path)
    test.to_csv(test_out_path + '/total.txt',columns = ['id', 'uid', 'mid', 'total'], index = False,sep='\t',header = None)
    test.to_csv(test_out_path + '/story.txt',columns = ['id', 'uid', 'mid', 'story'], index = False,sep='\t',header = None)
    test.to_csv(test_out_path + '/role.txt',columns = ['id', 'uid', 'mid', 'role'], index = False,sep='\t',header = None)
    test.to_csv(test_out_path + '/show.txt',columns = ['id', 'uid', 'mid', 'show'], index = False,sep='\t',header = None)
    test.to_csv(test_out_path + '/image.txt',columns = ['id', 'uid', 'mid', 'image'], index = False,sep='\t',header = None)
    test.to_csv(test_out_path + '/music.txt',columns = ['id', 'uid', 'mid', 'music'], index = False,sep='\t',header = None)    
    test.to_csv(test_out_path + '/all.txt',columns = ['id', 'uid', 'mid', 'total', 'story', 'role', 'show', 'image', 'music'], index = False,sep='\t',header = None)    



in_path = './tripadvisor(v1)/'
for k in range(1,21,1):
    split(k,in_path)