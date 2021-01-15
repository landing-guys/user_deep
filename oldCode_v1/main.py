import os
from time import perf_counter
from math import log2
import matplotlib.pyplot as plt

from surprise import Dataset
from surprise import Reader
from surprise import KNNWithMeans
from surprise import accuracy
from surprise.model_selection import train_test_split
from surprise.prediction_algorithms import predictions


start = perf_counter()
def savePred(file, predictions):
    if os.path.exists(file):
        os.remove(file)
    for p in predictions:
        with open(file, 'a+') as f:
            f.write(p[0] + '\t' + p[1] + '\t' + str(p[2]) + '\t' + str(p[3]) + '\n')
            
def getH(H, P):
    n_users, all_userid = computeN_users(P)
    for uid in all_userid:
        H.append((uid,computeUid_H(uid, P)))
        
def computeN_users(rates_matrix):
    n_users = 0
    all_userid = []
    for i in rates_matrix:
        if i[0] not in all_userid:
            all_userid.append(i[0])
    n_users = len(all_userid)
    return n_users, all_userid

def computeUid_H(uid, rates_matrix):
    H = 0
    P = getUid_P(uid, rates_matrix)
    for p in P:
        H = H + -log2(p)*p
    return H

def getUid_P(uid, rates_matrix):
    totalrates = 0
    P= []
    for i in rates_matrix:
        if i[0] == uid:
            totalrates = totalrates + i[3]
    for i in rates_matrix:
        if i[0] == uid:
            P.append(i[3]/totalrates)
    return P    

#将各个准则的预测分合并，方便计算
def get_P(P, story_P, role_P, show_P, image_P, music_P):
    for i in range(0, len(story_P)):
         P.append((story_P[i][0], story_P[i][1], story_P[i][3], role_P[i][3], show_P[i][3], image_P[i][3], music_P[i][3]))
#计算基于信息熵的多准则预测分
def entropy_based_pred(H_pred, H, P):
    for h in H:
        userid = h[0]
        total_h = h[1] + h[2] + h[3] + h[4] + h[5]
        multi_rate = 0
        if total_h > 0:
            for p in P:
                if p[0] == userid:
                    multi_rate = (p[2]*h[1] + p[3]*h[2] + p[4]*h[3] + p[5]*h[4] + p[6]*h[5])/total_h
                    H_pred.append((p[0], p[1], multi_rate))
        else:
            for p in P:
                if p[0] == userid:
                    multi_rate = (p[2] + p[3] + p[4] + p[5] + p[6])/5
                    H_pred.append((p[0], p[1], multi_rate))
#将计算出来的多准则预测分转换成Prediction类，方便计算
def transtoPredictions(multi_P, total_P, P):
    for m in multi_P:
        for t in total_P:
            if m[0] == t[0] and m[1] == t[1]:
                p = predictions.Prediction(m[0], m[1], t[2], m[2], t[4])
                P.append(p)
            
            
def shows(baseline, multi, s):
    x = [k for k in range(5,35,5)]
    fig = plt.figure()
    plt.grid(axis="y")
    plt.plot(x, baseline, label='baseline')
    plt.plot(x, multi, label='our_multi')
    plt.xlabel('No.of Neighbors')
    plt.ylabel(s)
    plt.title('Baseline vs Information entropy based multi-criteria')
    fig.legend()
    fig.show()
    

# read Datasets
file_path = 'D:/pythonProject/multi_criteria_RS/yahooDatas/datas4-33193/'
reader = Reader(line_format='timestamp item user rating', sep='\t')#timestamp
#载入数据，包括多准则评分：故事，角色，表演，画面，音乐，以及整体评分
story = Dataset.load_from_file(file_path + 'story.txt', reader=reader)
role = Dataset.load_from_file(file_path + 'role.txt', reader=reader)
show = Dataset.load_from_file(file_path + 'show.txt', reader=reader)
image = Dataset.load_from_file(file_path + 'image.txt', reader=reader)
music = Dataset.load_from_file(file_path + 'music.txt', reader=reader)
total = Dataset.load_from_file(file_path + 'total.txt', reader=reader)
print('载入数据成功！\n')
#按2:8拆分数据
random_states = 180
story_train, story_test = train_test_split(story, random_state = random_states)
role_train, role_test = train_test_split(role, random_state = random_states)
show_train, show_test = train_test_split(show, random_state = random_states)
image_train, image_test = train_test_split(image, random_state = random_states)
music_train, music_test = train_test_split(music, random_state = random_states)
total_train, total_test = train_test_split(total, random_state = random_states)
print('数据划分成功！\n')
#选择的是基于项目的协同过滤算法，项目相似度计算采用cosine方法
sim_options = {'name': 'pearson',#用皮尔森基线相似度避免出现过拟合
               'user_based': True} # 基于用户的协同过滤算法
baseline_mae, multi_mae = [],[]
baseline_rmse, multi_rmse = [],[]
# for k in range(5,35,5):
#     algo1 = KNNWithMeans(k,sim_options=sim_options)
#     algo2 = KNNWithMeans(k,sim_options=sim_options)
#     algo3 = KNNWithMeans(k,sim_options=sim_options)
#     algo4 = KNNWithMeans(k,sim_options=sim_options)
#     algo5 = KNNWithMeans(k,sim_options=sim_options)
#     algo6 = KNNWithMeans(k,sim_options=sim_options)
#     #训练数据（计算相似度）
#     algo1.fit(story_train)
#     algo2.fit(role_train)
#     algo3.fit(show_train)
#     algo4.fit(image_train)
#     algo5.fit(music_train)
#     algo6.fit(total_train)
#     print('数据训练完毕！')
#     #测试集计算预测分
#     story_predictions = algo1.test(story_test)
#     role_predictions = algo2.test(role_test)
#     show_predictions = algo3.test(show_test)
#     image_predictions =algo4.test(image_test)
#     music_predictions = algo5.test(music_test)
#     total_predictions = algo6.test(total_test)
#     print('计算各准则预测分完毕!\n')
#     baseline_mae.append(accuracy.mae(total_predictions))
#     baseline_rmse.append(accuracy.rmse(total_predictions))
#     #计算用户信息熵
#     story_H, role_H, show_H, image_H, music_H = [], [], [], [], []
#     getH(story_H, story_predictions)
#     getH(role_H, role_predictions)
#     getH(show_H, show_predictions)
#     getH(image_H, image_predictions)
#     getH(music_H, music_predictions)
#     H = []
#     for i in range(0, len(story_H)):
#         H.append((story_H[i][0],story_H[i][1],role_H[i][1],show_H[i][1],image_H[i][1],music_H[i][1]))
#     #计算基于信息熵的多准则预测分
#     P = []
#     get_P(P, story_predictions, role_predictions, show_predictions, image_predictions, music_predictions)
#     final_pred = []
#     entropy_based_pred(final_pred, H, P)
#     #计算度量指标
#     multi_P = []
#     transtoPredictions(final_pred, total_predictions, multi_P)
#     multi_mae.append(accuracy.mae(multi_P))
#     multi_rmse.append(accuracy.rmse(multi_P))
# shows(baseline_mae, multi_mae, 'MAE')
# shows(baseline_rmse, multi_rmse, 'RMSE')
dur = perf_counter() - start
print('程序运行时间为{:5f}s'.format(dur))



