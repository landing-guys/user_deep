# -*- coding: utf-8 -*-
"""
Created on Sat Sep 26 15:45:47 2020

@author: Jerry
"""


from ExperimentDatas import EDatas

readpath = './Datas/alldata-188854.txt'
writepath = 'yahoodata(interval-1)'
sdatas = EDatas(no_of_criteria = 5, steps = 4, ways = 'interval')
sdatas.readDatas(readpath)
sdatas.splitDatas(writepath)