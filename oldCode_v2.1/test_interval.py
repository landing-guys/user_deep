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

def plot_mae_increase(x, maes):
    increase = []
    for m in maes:
        increase.append(((m[0]-m[1]) / m[0])*100)
    fig = plt.figure()
    plt.plot(x, increase, label='算法提升率')
    plt.grid(axis="y")
    plt.xlabel('用户度范围')
    plt.ylabel('算法提升率')
    plt.title('不同用户度范围下多准则算法相较于单准则算法mae提升率')
    fig.legend()
    # fig.savefig(filepath)
    fig.show()

dirs = './yahoodata(interval-2)/'
readpath = os.listdir(dirs)
readpath.sort(key = lambda x:int(x.split('-')[0]))
maes = []
ks = [i for i in range(1, 22)] 
for i in range(len(readpath)):
    test = Experiment(no_of_criteria = 5, combin_func = 'total_reg')
    test.readDatas(dirs + readpath[i]+'/all.txt')
    s_mae, m_mae = test.SVD()
    maes.append((s_mae, m_mae))
plot_mae(readpath, maes)
plot_mae_increase(readpath, maes)
 



    