import os
from experiment import Experiment
import matplotlib.pyplot as plt

def plot_mae(x, maes):
    plt.rcParams['axes.unicode_minus']=False
    plt.rcParams['font.sans-serif']=['SimHei']
    y1,y2 = [], []
    for m in maes:
        y1.append(m[0])
        y2.append(m[1])
    fig = plt.figure()
    plt.grid(axis="y")
    plt.plot(x, y1, label='单准则')
    plt.plot(x, y2, label='多准则')
    plt.xlabel('用户度范围')
    plt.ylabel('mae')
    plt.title('不同用户度范围下多准则算法相较于单准则算法mae值')
    fig.legend()
    # fig.savefig(filepath)
    fig.show() 

def plot_mae_increase(x, maes, path):
    plt.rcParams['axes.unicode_minus']=False
    plt.rcParams['font.sans-serif']=['SimHei']
    y1,y2 = [], []
    for m in maes:
        y1.append(m[0])
        y2.append(m[1])
    fig = plt.figure()
    plt.grid(axis="y")
    plt.plot(x, y1, label='小度')
    plt.plot(x, y2, label='大度')
    plt.xlabel('用户度划分阈值')
    plt.ylabel('mae提升率')
    plt.title('不同用户度阈值下大度用户小度用户的mae提升率')
    fig.legend()
    fig.savefig(path)
    fig.show()

dirs = './yahoodata(step)/'
readpath = os.listdir(dirs)
readpath.sort(key = lambda x:int(x.split('-')[0]))
min_maes, max_maes = [], []
ks = [i for i in range(1, 22)] 
for i in range(len(readpath)):
    mins = Experiment(no_of_criteria = 5, combin_func = 'total_reg')
    mins.readDatas(dirs + readpath[i]+'/min/all.txt')
    maxs = Experiment(no_of_criteria = 5, combin_func = 'total_reg')
    maxs.readDatas(dirs + readpath[i]+'/more/all.txt')
    min_s_mae, min_m_mae = mins.SVD()
    max_s_mae, max_m_mae = maxs.SVD()
    min_maes.append((min_s_mae, min_m_mae))
    max_maes.append((max_s_mae, max_m_mae))
plot_mae(readpath, min_maes)
plot_mae(readpath, max_maes)
mae_increase = []
for i in range(len(min_maes)):
    min_increase = round(100*(min_maes[i][0]-min_maes[i][1]) / min_maes[i][0],4)
    max_increase = round(100*(max_maes[i][0]-max_maes[i][1]) / max_maes[i][0],4)
    mae_increase.append((min_increase, max_increase))
names = ['./(SVD)mae具体值.txt','./(SVD)mae提升率.txt','./(SVD)mae提升率对比图.png']

for n in names:
    if(os.path.exists(n)):
        os.remove(n)
for i in range(len(min_maes)):
    f1 = open(names[0],'a+')
    f1.write(str(i+1) + '\t' + str(min_maes[i][0]) + '\t' + str(min_maes[i][1]) + '\t' + str(max_maes[i][0])  + '\t' + str(max_maes[i][1]) +'\n')
    f1.close()
    f2 = open(names[1],'a+')
    f2.write(str(i+1) + '\t' + str(mae_increase[i][0]) + '\t' + str(mae_increase[i][1]) + '\n')
    f2.close()
plot_mae_increase(readpath, mae_increase, names[2])




    