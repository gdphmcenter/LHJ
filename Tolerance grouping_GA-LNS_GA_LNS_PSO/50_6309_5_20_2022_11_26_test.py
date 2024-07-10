import math
import argparse
import numpy as np
import torch
import datetime
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from gpn import GPN
import pandas as pd
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
from scipy.integrate import cumtrapz
import random
from scipy.stats import gaussian_kde

df = pd.read_excel('5_20_2022_11_26.xlsx')
# df = df.sort_values(by='序号', ascending=True)
m = 0
hetaorate_threshold = 0.9
stop_condition_met = False

while not stop_condition_met:  # 当stop_condition_met为False时，循环继续
    tour_len = 0
    hetao1 = []

    # 重新随机抽取50个不同的内沟径值和50个不同的外沟径值
    innerRaces = df['内沟径值'].sample(50)
    outerRaces = df['外沟径值'].sample(50)
    zuhe = []
    # 生成滚动体样本
    job_g_piancha = np.random.choice([-8, -6, -4, -2, 0, 2, 4], size=50, replace=True)
    # 组合内外沟径和滚动体样本
    zuhe.extend(outerRaces)
    zuhe.extend(innerRaces)
    zuhe.extend(job_g_piancha)
    # 将 NumPy 数组转换为 PyTorch 张量
    data_set = torch.tensor(zuhe, dtype=torch.float32).view(1, 150, 1)
    X = data_set

    time1 = datetime.datetime.now()
    # args
    parser = argparse.ArgumentParser(description="GPN test")
    parser.add_argument('--size', default=150, help="size of model")
    parser.add_argument('--batch_size', default=1, help='')
    parser.add_argument('--test_size', default=150, help="size of TSP")
    parser.add_argument('--test_steps', default=1, help='')
    args = vars(parser.parse_args())

    B = int(args['batch_size'])
    size = int(args['size'])
    test_size = int(args['test_size'])
    n_test = int(args['test_steps'])
    load_root = 'C:/Users/chongchong/Desktop/强化学习/论文相关/论文代码/gpn150wazhou_new.pt'


    print('=========================')
    print('prepare to test')
    print('=========================')


    def getset():
        data_set = []
        zuhe = []
        for l in range(1):
            for j in range(2):
                # 产生外圈、内圈
                if j == 0:
                    def read_excel_data(file_path, column_name):
                        df = pd.read_excel(file_path, engine='openpyxl')  # 使用openpyxl引擎读取Excel文件
                        data = df[column_name].values  # 提取指定列的数据
                        data = data.reshape(-1, 1)  # 将数据转换为二维数组，以匹配sklearn的输入要求
                        return data
                    # 自定义评分函数
                    def my_scores(estimator, X):
                        scores = estimator.score_samples(X)
                        scores = scores[scores != -np.inf]  # 移除-inf值
                        return np.mean(scores)
                    # 网格搜索参数
                    kernels = ['gaussian']
                    h_vals = np.arange(0.05, 1, 0.1)
                    # Excel文件路径和列名
                    file_path = '5_20_2022_11_26.xlsx'
                    inner_column_name = '内沟径值'
                    outer_column_name = '外沟径值'
                    df = pd.read_excel(file_path, engine='openpyxl')
                    innerRaces = df['内沟径值'].sample(50, random_state=1)
                    outerRaces = df['外沟径值'].sample(50, random_state=1)
                    innerRaces = np.array(innerRaces)
                    outerRaces = np.array(outerRaces)
                    innerRaces = innerRaces.reshape(-1, 1)
                    outerRaces = outerRaces.reshape(-1, 1)
                    # 读取内沟径值数据并拟合KDE
                    x_train_inner = innerRaces
                    kde_inner = KernelDensity()
                    grid_inner = GridSearchCV(kde_inner,
                                              {'bandwidth': h_vals, 'kernel': kernels},
                                              scoring=my_scores,
                                              cv=5)  # 使用5折交叉验证
                    grid_inner.fit(x_train_inner)
                    best_kde_inner = grid_inner.best_estimator_

                    # 读取外沟尺寸数据并拟合KDE
                    x_train_outer = outerRaces = outerRaces.reshape(-1, 1)
                    kde_outer = KernelDensity()
                    grid_outer = GridSearchCV(kde_outer,
                                              {'bandwidth': h_vals, 'kernel': kernels},
                                              scoring=my_scores,
                                              cv=5)  # 使用5折交叉验证
                    grid_outer.fit(x_train_outer)
                    best_kde_outer = grid_outer.best_estimator_

                    # 生成测试数据
                    x_test_inner = np.linspace(x_train_inner.min(), x_train_inner.max(), 3000)[:, np.newaxis]
                    x_test_outer = np.linspace(x_train_outer.min(), x_train_outer.max(), 3000)[:, np.newaxis]
                    # print(x_test_inner)
                    # print(type(x_test_inner))
                    x_test_inner = x_test_inner.reshape(-1, 1)
                    x_test_outer = x_test_outer.reshape(-1, 1)

                    # 计算概率密度和CDF（内沟径值）
                    log_dens_inner = best_kde_inner.score_samples(x_test_inner)
                    prob_dens_inner = np.exp(log_dens_inner)
                    cdf_inner = cumtrapz(prob_dens_inner, x_test_inner[:, 0], initial=0)
                    cdf_inner = cdf_inner / cdf_inner[-1]  # 归一化

                    # 计算概率密度和CDF（外沟尺寸）
                    log_dens_outer = best_kde_outer.score_samples(x_test_outer)
                    prob_dens_outer = np.exp(log_dens_outer)
                    cdf_outer = cumtrapz(prob_dens_outer, x_test_outer[:, 0], initial=0)
                    cdf_outer = cdf_outer / cdf_outer[-1]  # 归一化

                    # 设定目标概率值
                    target_probs = [random.random() for _ in range(50)]

                    # 查找内沟径值对应的横坐标值
                    result_x_inner = []
                    for prob in target_probs:
                        idx = np.searchsorted(cdf_inner, prob, side='right') - 1
                        if idx < 0:
                            result_x_inner.append(round(x_test_inner[0, 0], 2))
                        elif idx >= len(cdf_inner) - 1:
                            result_x_inner.append(round(x_test_inner[-1, 0], 2))
                        else:
                            result_x_inner.append(round(x_test_inner[idx, 0], 2))

                    # 查找外沟尺寸对应的横坐标值
                    result_x_outer = []
                    for prob in target_probs:
                        idx = np.searchsorted(cdf_outer, prob, side='right') - 1
                        if idx < 0:
                            result_x_outer.append(round(x_test_outer[0, 0], 2))
                        elif idx >= len(cdf_outer) - 1:
                            result_x_outer.append(round(x_test_outer[-1, 0], 2))
                        else:
                            result_x_outer.append(round(x_test_outer[idx, 0], 2))

                    zuhe.extend(result_x_outer)
                    zuhe.extend(result_x_inner)
                    print(result_x_outer)
                    print(result_x_inner)


                # 产生滚动体
                if j == 1:
                    job_g_piancha = list(np.random.choice([-8, -6, -4, -2, 0, 2, 4], size=50, replace=True))
                    zuhe.extend(job_g_piancha)
        zuhe = np.array(zuhe)
        zuhe = zuhe.reshape(1,150,1)          # batch、size、value
        data_set = zuhe
        # print(data_set)
        return data_set

    # greedy
    model = torch.load(load_root)

    tour_len = 0
    total_len = 0
    tor_len_hetaolv = 0

    for m in range(n_test):
        tour_len = 0

        # X = getset()
        # X = torch.Tensor(X)
        X = X

        mask = torch.zeros(B, test_size)

        R = 0
        Idx = []
        reward = 0

        Y = X.view(B, test_size, 1)  # to the same batch size
        x = Y[:, 0, :]
        h = None
        c = None
        batch_list2 = []
        for k in range(test_size):
            output, h, c, _ = model(x=x, X_all=X, h=h, c=c, mask=mask)
            idx = torch.argmax(output, dim=1)
            Idx.append(idx.data)
            # Y1 = Y[[i for i in range(B)], idx.data].clone()
            # if k == 0:
            #     Y_ini = Y1.clone()
            # if k > 0:
            #     reward = torch.norm(Y1-Y0, dim=1)
            #
            # Y0 = Y1.clone()
            # x = Y[[i for i in range(B)], idx.data].clone()
            #
            # R += reward

            # 游隙计算
            batch_list2.extend(idx)
            mask[[i for i in range(B)], idx.data] += -np.inf

        youxi_num_batch2 = []
        hetao1 = []
        for j in range(B):
            batch_reward_j = []
            for Q in range(size):  # size:150
                batch_reward_j.append(batch_list2[1 * Q + j])
            w_j = []
            n_j = []
            g_j = []
            for r in batch_reward_j:
                if r < torch.tensor(50):
                    w_j.append(r)
                if torch.tensor(49) < r < torch.tensor(100):
                    n_j.append(r)
                if r > torch.tensor(99):
                    g_j.append(r)
            batch_j_input = Y[j, :, :]
            batch_j_input = batch_j_input.view(150)
            youxi = 0
            T = 0
            RRR = []
            for t in range(50):
                youxi_true = (batch_j_input[int(w_j[t])] - batch_j_input[int(n_j[t])] - 2 * batch_j_input[int(g_j[t])])
                RRR.append(youxi_true)
                youxi += torch.tensor(math.pow(abs(torch.tensor(12.5) - youxi_true), 2))
                # youxi += abs(torch.tensor(25) - youxi_true)
                if torch.tensor(5) < youxi_true < torch.tensor(20):
                    T += 1
            # print(RRR)
            hetao1.append(T)
            youxi = youxi / torch.tensor(50)
            youxi_num_batch2.append(youxi)
        R = torch.tensor(youxi_num_batch2)

        # # 查组
        # n_j_50 = []
        # g_j_50 = []
        # for R in n_j:
        #     n_j_50.append(R-torch.tensor(50))
        # for U in g_j:
        #     g_j_50.append(U-torch.tensor(100))

        tour_len += R.mean().item()

        current_hetao_rate = sum(he tao1) / len(hetao1) / 50

        # 打印当前测试的合套率
        print(f'test:{m}, 平均游隙:{tour_len}, 平均合套率:{current_hetao_rate}')

        # 更新总游隙和总合套率
        total_len += tour_len
        tor_len_hetaolv += current_hetao_rate

        # 检查合套率是否大于0.8，如果是，则停止循环
        if current_hetao_rate > hetaorate_threshold:
            stop_condition_met = True
        else:
            m += 1  # 如果没有达到停止条件，更新测试步


time2 = datetime.datetime.now()
print('total 游隙均值:', total_len / n_test, "total 平均合套率：", tor_len_hetaolv / n_test)
print('running time_batch_64:', (time2 - time1).seconds, "秒")
print('running time_batch_1:', ((time2 - time1).seconds) / 64, "秒")
