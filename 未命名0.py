import os
from RecSys import Experiment
import matplotlib.pyplot as plt

def plot_mae(x, maes, filepath, strs):
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
    plt.title(strs)
    fig.legend()
    fig.savefig(filepath)
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


E = Experiment(no_of_criteria = 5)
split_options = {'k_item':10,'k_user':5,'random_state':666}
E.read_split_Datas('./Datas/yahoo.txt', split_options)
E.SlopeOne_train()
pred_option = {'is_total':False, 'combin_func':'info_entropy'}
E.Predict(pred_options = pred_option)
maes = []