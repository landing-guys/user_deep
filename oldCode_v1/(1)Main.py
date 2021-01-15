import os
import numpy as np
from time import perf_counter
from math import log2
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

from surprise import Dataset
from surprise import Reader
from surprise import KNNWithMeans
from surprise import accuracy
from surprise.model_selection import train_test_split
from surprise.prediction_algorithms import predictions

from sklearn.linear_model import LinearRegression

def totalRegModel(df):
    # data_train = np.array(df[['story','role','show','image','music']]).reshape(len(df),5)
    data_train = np.array(df[['xinJiaBi', 'shuShiDu', 'weiZi', 'weiShen', 'fuWu']]).reshape(len(df),5)
    data_test = np.array(df['total']).reshape(len(df),1) 
    #整个数据集的回归模型
    reg = LinearRegression()
    reg.fit(data_train,data_test)
    k = reg.coef_[0]
    b = reg.intercept_[0]
    return k, b

def totalReg(P, k, b, single_p):
    multi_p = []
    for i in range(len(P)):
        pred = P[i][3]*k[0] + P[i][4]*k[1] + P[i][5]*k[2] + P[i][6]*k[3] + P[i][7]*k[4] + b
        multi_p.append((P[i][0], P[i][1], pred))
    P = transtoPrediction(multi_p, single_p)
    return P

def userRegModel(df):
    c = Counter(df.uid).most_common()
    uid = []
    data = np.array(c)
    param = []
    #加入全局回归模型，在没有该用户的回归模型时
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

# def userReg(P, param, single_p):
#     multi_p = []
    
#     for k in param:
#     for i in range(len(P)):

        
        
def avg(P, single_p):
    multi_p = []
    for i in range(len(P)):
        pred = (P[i][3] + P[i][4] + P[i][5] + P[i][6] + P[i][7] + P[i][2])/6
        multi_p.append((P[i][0], P[i][1], pred))
    P = transtoPrediction(multi_p, single_p)
    return P



def transtoPrediction(multi, single):
    P = []
    for m in multi:
        for s in single:
            if m[0] == s[0] and m[1] == s[1]:
                p = predictions.Prediction(m[0], m[1], s[2], m[2], s[4])
                P.append(p)
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

def combine(p1, p2, p3, p4, p5, p6):
    P = []
    for i in range(len(p1)):
        P.append((p6[i].uid, p6[i].iid, p6[i].est, p1[i].est,p2[i].est,p3[i].est,p4[i].est,p5[i].est))
    return P

def mian(k,datasname):
    #阈值为1的小度数据集实验
    d_mae, x_mae = [],[]
    # d_rmse, x_rmse = [],[]
    file_path = './' + datasname + '/' + str(k) 
    # file_path = './' + str(k) 
    x_mae = chose_tripadvisor(file_path + '/min/')
    d_mae = chose_tripadvisor(file_path + '/more/')
    return x_mae, d_mae # d_rmse, x_rmse

def chose_tripadvisor(file_path):
    # mae= []
    # rmse = []
    reader = Reader(line_format='timestamp user item  rating', sep='\t')#timestamp
    #载入数据，包括多准则评分：故事，角色，表演，画面，音乐，以及整体评分
    xinjiabi = Dataset.load_from_file(file_path + 'xinJiaBi.txt', reader=reader)
    shushidu = Dataset.load_from_file(file_path + 'shuShiDu.txt', reader=reader)
    weizi = Dataset.load_from_file(file_path + 'weiZi.txt', reader=reader)
    weiShen = Dataset.load_from_file(file_path + 'weiShen.txt', reader=reader)
    # shuimian = Dataset.load_from_file(file_path + 'shuiMian.txt', reader=reader)
    fuwu = Dataset.load_from_file(file_path + 'fuWu.txt', reader=reader)
    total = Dataset.load_from_file(file_path + 'total.txt', reader=reader)
    # print('载入数据成功！\n')
    #按2:8拆分数据
    random_states = 180
    xinjiabi_train, xinjiabi_test = train_test_split(xinjiabi, random_state = random_states)
    shushidu_train, shushidu_test = train_test_split(shushidu, random_state = random_states)
    weizi_train, weizi_test = train_test_split(weizi, random_state = random_states)
    weiShen_train, weiShen_test = train_test_split(weiShen, random_state = random_states)
    # shuimian_train, shuimian_test = train_test_split(shuimian, random_state = random_states)
    fuwu_train, fuwu_test = train_test_split(fuwu, random_state = random_states)
    total_train, total_test = train_test_split(total, random_state = random_states)
    # print('数据划分成功！\n')
    #选择的是基于项目的协同过滤算法，项目相似度计算采用cosine方法
    sim_options = {'name': 'pearson',#用皮尔森基线相似度避免出现过拟合
                   'user_based': False} # 基于用户的协同过滤算法
    algo1 = KNNWithMeans(sim_options=sim_options)
    algo2 = KNNWithMeans(sim_options=sim_options)
    algo3 = KNNWithMeans(sim_options=sim_options)
    algo4 = KNNWithMeans(sim_options=sim_options)
    # algo5 = KNNWithMeans(sim_options=sim_options)
    algo6 = KNNWithMeans(sim_options=sim_options)
    algo7 = KNNWithMeans(sim_options=sim_options)
    algo1.fit(xinjiabi_train)
    algo2.fit(shushidu_train)
    algo3.fit(weizi_train)
    algo4.fit(weiShen_train)
    # algo5.fit(shuimian_train)
    algo6.fit(fuwu_train)
    algo7.fit(total_train)
    xinjiabi_p = algo1.test(xinjiabi_test)
    shushidu_p = algo2.test(shushidu_test)
    weizi_p = algo3.test(weizi_test)
    weishen_p =algo4.test(weiShen_test)
    # shuimian_p = algo5.test(shuimian_test)
    fuwu_p = algo6.test(fuwu_test)
    total_p = algo7.test(total_test)
    # rmse.append(accuracy.rmse(single_p))
    #平均法
    P = combine(xinjiabi_p, shushidu_p, weizi_p, weishen_p, fuwu_p, total_p)
    # multi_p = avg(P, total_p)
    #整体回归
    df = pd.read_csv(file_path + 'all.txt', sep = '\t', names = ['id', 'uid', 'hid', 'total', 'xinJiaBi', 'shuShiDu', 'weiZi', 'weiShen', 'shuiMian', 'fuWu'])    
    df = df[['id', 'uid', 'hid', 'total', 'xinJiaBi', 'shuShiDu', 'weiZi', 'weiShen', 'fuWu']]
    k, b = totalRegModel(df)
    multi_p = totalReg(P, k, b, total_p)
    
    #基于每个用户的回归
    
    mae = (accuracy.mae(total_p),accuracy.mae(multi_p))
    # rmse.append(accuracy.rmse(multi_p))
    return mae#, rmse
   

def chose_yahoo(file_path):
    # mae= []
    # rmse = []
    reader = Reader(line_format='timestamp user item  rating', sep='\t')#timestamp
    #载入数据，包括多准则评分：故事，角色，表演，画面，音乐，以及整体评分
    story = Dataset.load_from_file(file_path + 'story.txt', reader=reader)
    role = Dataset.load_from_file(file_path + 'role.txt', reader=reader)
    show = Dataset.load_from_file(file_path + 'show.txt', reader=reader)
    image = Dataset.load_from_file(file_path + 'image.txt', reader=reader)
    music = Dataset.load_from_file(file_path + 'music.txt', reader=reader)
    total = Dataset.load_from_file(file_path + 'total.txt', reader=reader)
    # print('载入数据成功！\n')
    #按2:8拆分数据
    random_states = 180
    story_train, story_test = train_test_split(story, random_state = random_states)
    role_train, role_test = train_test_split(role, random_state = random_states)
    show_train, show_test = train_test_split(show, random_state = random_states)
    image_train, image_test = train_test_split(image, random_state = random_states)
    music_train, music_test = train_test_split(music, random_state = random_states)
    total_train, total_test = train_test_split(total, random_state = random_states)
    # print('数据划分成功！\n')
    #选择的是基于项目的协同过滤算法，项目相似度计算采用cosine方法
    sim_options = {'name': 'pearson',#用皮尔森基线相似度避免出现过拟合
                   'user_based': False} # 基于用户的协同过滤算法
    algo1 = KNNWithMeans(sim_options=sim_options)
    algo2 = KNNWithMeans(sim_options=sim_options)
    algo3 = KNNWithMeans(sim_options=sim_options)
    algo4 = KNNWithMeans(sim_options=sim_options)
    algo5 = KNNWithMeans(sim_options=sim_options)
    algo6 = KNNWithMeans(sim_options=sim_options)
    algo1.fit(story_train)
    algo2.fit(role_train)
    algo3.fit(show_train)
    algo4.fit(image_train)
    algo5.fit(music_train)
    algo6.fit(total_train)
    story_p = algo1.test(story_test)
    role_p = algo2.test(role_test)
    show_p = algo3.test(show_test)
    image_p =algo4.test(image_test)
    music_p = algo5.test(music_test)
    single_p = algo6.test(total_test)
    # rmse.append(accuracy.rmse(single_p))
    #平均法
    # multi_p = avg(story_p, role_p, show_p, image_p, music_p, single_p)
    #整体回归
    P = combine(story_p, role_p, show_p, image_p, music_p, single_p)
    df = pd.read_csv(file_path + 'all.txt', sep = '\t', names = ['id', 'uid', 'mid', 'total', 'story', 'role', 'show', 'image', 'music'])
    k, b = totalRegModel(df)
    multi_p = totalReg(P, k, b, single_p)
    #基于每个用户的回归
    
    mae = (accuracy.mae(single_p),accuracy.mae(multi_p))
    # rmse.append(accuracy.rmse(multi_p))
    return mae#, rmse

mae = []
maes = []
datasname = 'tripadvisor'
for k in range(1,41,1):
    x_mae,d_mae = mian(k,datasname)
    t1 = (x_mae[1]-x_mae[0])/x_mae[0] 
    t2 = (d_mae[1]-d_mae[0])/d_mae[0]
    print('阈值为{0}时小度数据下多准则算法精度提升率为{1}%'.format(k,-t1*100))
    print('阈值为{0}时大度数据下多准则算法精度提升率为{1}%'.format(k,-t2*100))
    mae.append((k, -t1*100, -t2*100))
    maes.append((k, x_mae[0], x_mae[1], d_mae[0], d_mae[1]))
f = './实验结果/' + datasname 
if not os.path.exists(f):
        os.makedirs(f)
names = ['/(包含整体评分)平均法mae.txt','/(包含整体评分)平均法mae提升率.txt','/(包含整体评分)平均法mae提升率对比图.png']
fs = []
for n in names:
    fs.append(f+n)
for f in fs:
    if(os.path.exists(f)):
        os.remove(f)
for m in maes:
    f1 = open(fs[0],'a+')
    f1.write(str(m[0]) + '\t' + str(m[1]) + '\t\t' + str(m[2])  + '\t\t' + str(m[3])  + '\t\t' + str(m[4]) +'\n')
    f1.close()
for m in mae:
    f2 = open(fs[1],'a+')
    f2.write(str(m[0]) + '\t' + str(m[1]) + '\t\t' + str(m[2]) +'\n')
    f2.close()
shows(mae, fs[2])
    
 
    
    
    
# #阈值为1的小度数据集实验
# single_mae, single_rmse = [],[]
# multi_mae, multi_rmse = [],[]
# file_path = './' + '20' + '/min/'
# reader = Reader(line_format='timestamp item user rating', sep='\t')#timestamp
# #载入数据，包括多准则评分：故事，角色，表演，画面，音乐，以及整体评分
# story = Dataset.load_from_file(file_path + 'story.txt', reader=reader)
# role = Dataset.load_from_file(file_path + 'role.txt', reader=reader)
# show = Dataset.load_from_file(file_path + 'show.txt', reader=reader)
# image = Dataset.load_from_file(file_path + 'image.txt', reader=reader)
# music = Dataset.load_from_file(file_path + 'music.txt', reader=reader)
# total = Dataset.load_from_file(file_path + 'total.txt', reader=reader)
# print('载入数据成功！\n')
# #按2:8拆分数据
# random_states = 180
# story_train, story_test = train_test_split(story, random_state = random_states)
# role_train, role_test = train_test_split(role, random_state = random_states)
# show_train, show_test = train_test_split(show, random_state = random_states)
# image_train, image_test = train_test_split(image, random_state = random_states)
# music_train, music_test = train_test_split(music, random_state = random_states)
# total_train, total_test = train_test_split(total, random_state = random_states)
# print('数据划分成功！\n')
# #选择的是基于项目的协同过滤算法，项目相似度计算采用cosine方法
# sim_options = {'name': 'pearson',#用皮尔森基线相似度避免出现过拟合
#                'user_based': True} # 基于用户的协同过滤算法
# algo1 = KNNWithMeans(sim_options=sim_options)
# algo2 = KNNWithMeans(sim_options=sim_options)
# algo3 = KNNWithMeans(sim_options=sim_options)
# algo4 = KNNWithMeans(sim_options=sim_options)
# algo5 = KNNWithMeans(sim_options=sim_options)
# algo6 = KNNWithMeans(sim_options=sim_options)
# algo1.fit(story_train)
# algo2.fit(role_train)
# algo3.fit(show_train)
# algo4.fit(image_train)
# algo5.fit(music_train)
# algo6.fit(total_train)
# story_p = algo1.test(story_test)
# role_p = algo2.test(role_test)
# show_p = algo3.test(show_test)
# image_p =algo4.test(image_test)
# music_p = algo5.test(music_test)
# single_p = algo6.test(total_test)
# single_mae.append(accuracy.mae(single_p))
# single_rmse.append(accuracy.rmse(single_p))
# multi_p = avg(story_p, role_p, show_p, image_p, music_p, single_p)
# single_mae.append(accuracy.mae(multi_p))
# single_rmse.append(accuracy.rmse(multi_p))
        







