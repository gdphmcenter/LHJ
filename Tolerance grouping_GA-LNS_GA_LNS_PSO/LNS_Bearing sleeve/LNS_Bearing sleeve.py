import random
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
import datetime
import math
from sympy import *
import scipy.stats as st
import copy


class Solution:
    # 使用符号编号表示一个访问路径route
    def __init__(self):
        self.route = []  # 对应分组的编码（染色体）
        self.cost = 0  # 解对应的总成本
        self.pro = []  # 对应分组的概率

class Lns_tsp:
    def __init__(self):
        self.a = 1
    def get_route_cost(self, genes):
        # 根据分组计算游隙和合套率成本
        time1 = datetime.datetime.now()
        num_zu = 0  # 组数
        genes_ = np.random.randint(0, 2, 10).tolist()
        zero_hot = genes_  # 0/1编码确定组数，随机概率
        w_zu = genes[:10]  # 外圈组
        n_zu = genes[10:20]  # 内圈组
        g_zu = genes[20:30]  # 滚动体组
        for s in zero_hot:
            if s == 1:
                num_zu += 1
            else:
                pass
        zhen_9_1 = []
        if num_zu == 1:
            pro = [1.]
        elif num_zu == 0:
            pro = []
        else:
            for O in range(10000):
                pro_0 = np.random.dirichlet(np.ones(num_zu), size=1)  # 随机概率
                xiaxian = min(pro_0[0])
                shangxian = max(pro_0[0])
                if xiaxian > 0.02 and shangxian < 0.98:
                    zhen_9_1.append(pro_0)
                    break
            pro = random.choice(zhen_9_1)
            pro = pro[0]
        # print(pro)
        t_w = []
        t_n = []
        t_g = []
        for h in w_zu:
            if h < num_zu:
                t_w.append(h)
            else:
                pass
        for p in n_zu:
            if p < num_zu:
                t_n.append(p)
            else:
                pass
        for j in g_zu:
            if j < num_zu:
                t_g.append(j)
            else:
                pass
            # 外圈、内圈、滚动体概率
        pro_w = []  # 已知组的个数，外圈的组概率
        pro_n = []  # 已知组的个数，内圈的组概率
        pro_g = []  # 已知组的个数，滚动体的组概率
        for L in t_w:
            pro_w.append(pro[L])
        for p in t_n:
            pro_n.append(pro[p])
        for o in t_g:
            pro_g.append(pro[o])
            # 外圈均值及方差计算
        ################################################################ 外圈换算具体的偏差值
        pro_piancha = []
        pro_piancha_n = []
        pro_piancha_g = []
        xiaxian = []
        w_shuzhi = 0
        # y = 0
        time2 = datetime.datetime.now()
        # print("概率划分运行时间_1：", (time2 - time1).seconds)
        ########################################################################################
        strattime = datetime.datetime.now()  # 开始时间

        if len(pro_w) > 1:
            for P in range(1, len(pro_w)):
                pro_piancha.append(st.norm.ppf(sum(pro_w[:P]), loc=308.92, scale=0.00500))
                pro_piancha_n.append(st.norm.ppf(sum(pro_n[:P]), loc=244.745, scale=0.01333))
                pro_piancha_g.append(st.norm.ppf(sum(pro_g[:P]), loc=31.9925, scale=0.00417))

        endtime = datetime.datetime.now()
        # print("概率划分运行时间_2：",(endtime-strattime).seconds)
        length = len(pro_w)
        length_1 = len(pro_n)
        length_2 = len(pro_g)
        h = 0
        time3 = datetime.datetime.now()
        for z in range(11):  # 0-10 闭区间                  # 统一返回一个变量，再在最后用return
            if z == 0 and z == length:  # 0组
                fitness_11 = 999999999  # 可以给一个很大的值，来淘汰掉
                yyxx = []
            if z == 1 and z == length:  # 1组  全部合套，没有被排除的，只考虑合套游隙  一组的合套效果都不咋好
                fitness_1 = 0
                hetao_T = 0
                hetao_F = 0
                groud_0_w = w
                groud_0_n = n
                groud_0_g = g
                yyxx = []
                for r in range(len(w)):  # 优化目标：1.合套率（1.1数量原因没合成 1.2不符合游隙） 2.游隙均值  3.数据的方差
                    youxi = groud_0_w[r] - groud_0_n[r] - 2 * groud_0_g[r]
                    if 0.16 < youxi < 0.22:
                        hetao_T += 1
                        yyxx.append(youxi)
                    else:
                        hetao_F += 1  # 优化1：不符合合套游隙
                    chazhi = math.pow(abs(youxi - 0.19), 2)  # 与最佳游隙的差值
                    fitness_1 += chazhi
                fitness_1 = fitness_1 / len(w)  # 优化2：游隙均值
                fitness_11 = fitness_1 * 100000 + hetao_F
                # print("适应度", fitness_11, "目标：", yyxx, len(yyxx), hetao_T)
            if z == 2 and z == length:  # 2组          # 优化目标：1.合套率（1.1数量原因没合成 1.2不符合游隙） 2.游隙均值
                list_w = []
                list_n = []
                list_g = []
                groud_0_w = []  # 4
                groud_1_w = []  # 23
                groud_0_n = []  # 20
                groud_1_n = []  # 6
                groud_0_g = []  # 5
                groud_1_g = []  # 21
                for t in w:
                    if t < pro_piancha[0]:
                        groud_0_w.append(t)
                    if pro_piancha[0] < t:
                        groud_1_w.append(t)

                for u in n:
                    if u < pro_piancha_n[0]:
                        groud_0_n.append(u)
                    if pro_piancha_n[0] < u:
                        groud_1_n.append(u)
                for h in g:
                    if h < pro_piancha_g[0]:
                        groud_0_g.append(h)
                    if pro_piancha_g[0] < h:
                        groud_1_g.append(h)
                list_w.append(groud_0_w), list_w.append(groud_1_w)
                list_n.append(groud_0_n), list_n.append(groud_1_n)
                list_g.append(groud_0_g), list_g.append(groud_1_g)
                # 以滚动体为对照排序 即滚动体 ggg = [0,1,2,3,4,5]
                nnn = []  # 内圈
                www = []  # 外圈
                for r in pro_g:
                    T_N = 0
                    for p in pro_n:
                        if p != r:
                            T_N += 1
                        else:
                            nnn.append(T_N)
                    T_W = 0
                    for Z in pro_w:
                        if Z != r:
                            T_W += 1
                        else:
                            www.append(T_W)
                new_list_g = list_g
                new_list_n = []
                new_list_w = []
                for I in nnn:
                    new_list_n.append(list_n[I])
                for H in www:
                    new_list_w.append(list_w[H])
                #####################################################
                length_zu_0 = min(len(new_list_g[0]), len(new_list_n[0]), len(new_list_w[0]))
                length_zu_1 = min(len(new_list_g[1]), len(new_list_n[1]), len(new_list_w[1]))
                length_zu = length_zu_0 + length_zu_1
                defeat_1_1 = len(w) - length_zu
                new_new_list_g = []
                new_new_list_n = []
                new_new_list_w = []
                new_new_list_g.append(random.sample(new_list_g[0], length_zu_0)), new_new_list_g.append(
                    random.sample(new_list_g[1], length_zu_1))
                new_new_list_n.append(random.sample(new_list_n[0], length_zu_0)), new_new_list_n.append(
                    random.sample(new_list_n[1], length_zu_1))
                new_new_list_w.append(random.sample(new_list_w[0], length_zu_0)), new_new_list_w.append(
                    random.sample(new_list_w[1], length_zu_1))
                # fangcha_list_g = []
                # fangcha_list_n = []
                # fangcha_list_w = []
                # fangcha_list_g.append(new_new_list_g[:length_zu_0]), fangcha_list_g.append(new_new_list_g[length_zu_0:])
                # fangcha_list_n.append(new_new_list_n[:length_zu_0]), fangcha_list_n.append(new_new_list_n[length_zu_0:])
                # fangcha_list_w.append(new_new_list_w[:length_zu_0]), fangcha_list_w.append(new_new_list_w[length_zu_0:])

                new_new_list_g = sum(new_new_list_g, [])  # 列表扁平化处理
                new_new_list_n = sum(new_new_list_n, [])
                new_new_list_w = sum(new_new_list_w, [])
                fitness_1 = 0
                hetao_T = 0
                hetao_F = 0
                yyxx = []
                for r in range(len(new_new_list_g)):  # 优化目标：1.合套率（1.1数量原因没合成 1.2不符合游隙） 2.游隙均值
                    youxi = new_new_list_w[r] - new_new_list_n[r] - 2 * new_new_list_g[r]
                    if 0.16 < youxi < 0.22:
                        hetao_T += 1
                        yyxx.append(youxi)
                    else:
                        hetao_F += 1  # 优化1：不符合合套游隙
                    chazhi = math.pow(abs(youxi - 0.19), 2)  # 与最佳游隙的差值
                    fitness_1 += chazhi

                fitness_1 = fitness_1 / len(new_new_list_g)  # 优化2：游隙均值
                fitness_11 = fitness_1 * 100000 + hetao_F
                # print("适应度", fitness_11, "目标：", yyxx, len(yyxx), hetao_T)
            if z == 3 and z == length:
                list_w = []
                list_n = []
                list_g = []
                groud_0_w = []
                groud_1_w = []
                groud_2_w = []
                groud_0_n = []
                groud_1_n = []
                groud_2_n = []
                groud_0_g = []
                groud_1_g = []
                groud_2_g = []
                for t in w:
                    if t < pro_piancha[0]:
                        groud_0_w.append(t)
                    if pro_piancha[0] < t < pro_piancha[1]:
                        groud_1_w.append(t)
                    if pro_piancha[1] < t:
                        groud_2_w.append(t)
                for u in n:
                    if u < pro_piancha_n[0]:
                        groud_0_n.append(u)
                    if pro_piancha_n[0] < u < pro_piancha_n[1]:
                        groud_1_n.append(u)
                    if pro_piancha_n[1] < u:
                        groud_2_n.append(u)
                for h in g:
                    if h < pro_piancha_g[0]:
                        groud_0_g.append(h)
                    if pro_piancha_g[0] < h < pro_piancha_g[1]:
                        groud_1_g.append(h)
                    if pro_piancha_g[1] < h:
                        groud_2_g.append(h)
                list_w.append(groud_0_w), list_w.append(groud_1_w), list_w.append(groud_2_w)
                list_n.append(groud_0_n), list_n.append(groud_1_n), list_n.append(groud_2_n)
                list_g.append(groud_0_g), list_g.append(groud_1_g), list_g.append(groud_2_g)
                # 以滚动体为对照排序 即滚动体 ggg = [0,1,2,3,4,5]
                nnn = []  # 内圈
                www = []  # 外圈
                for r in pro_g:
                    T_N = 0
                    for p in pro_n:
                        if p != r:
                            T_N += 1
                        else:
                            nnn.append(T_N)
                    T_W = 0
                    for Z in pro_w:
                        if Z != r:
                            T_W += 1
                        else:
                            www.append(T_W)
                new_list_g = list_g
                new_list_n = []
                new_list_w = []
                for I in nnn:
                    new_list_n.append(list_n[I])
                for H in www:
                    new_list_w.append(list_w[H])
                #####################################################
                length_zu_0 = min(len(new_list_g[0]), len(new_list_n[0]), len(new_list_w[0]))
                length_zu_1 = min(len(new_list_g[1]), len(new_list_n[1]), len(new_list_w[1]))
                length_zu_2 = min(len(new_list_g[2]), len(new_list_n[2]), len(new_list_w[2]))
                length_zu = length_zu_0 + length_zu_1 + length_zu_2
                defeat_1_1 = len(w) - length_zu
                new_new_list_g = []
                new_new_list_n = []
                new_new_list_w = []
                new_new_list_g.append(random.sample(new_list_g[0], length_zu_0)), new_new_list_g.append(
                    random.sample(new_list_g[1], length_zu_1)), new_new_list_g.append(
                    random.sample(new_list_g[2], length_zu_2))
                new_new_list_n.append(random.sample(new_list_n[0], length_zu_0)), new_new_list_n.append(
                    random.sample(new_list_n[1], length_zu_1)), new_new_list_n.append(
                    random.sample(new_list_n[2], length_zu_2))
                new_new_list_w.append(random.sample(new_list_w[0], length_zu_0)), new_new_list_w.append(
                    random.sample(new_list_w[1], length_zu_1)), new_new_list_w.append(
                    random.sample(new_list_w[2], length_zu_2))

                new_new_list_g = sum(new_new_list_g, [])  # 列表扁平化处理
                new_new_list_n = sum(new_new_list_n, [])
                new_new_list_w = sum(new_new_list_w, [])
                fitness_1 = 0
                hetao_T = 0
                hetao_F = 0
                yyxx = []
                for r in range(len(new_new_list_g)):  # 优化目标：1.合套率（1.1数量原因没合成 1.2不符合游隙） 2.游隙均值
                    youxi = new_new_list_w[r] - new_new_list_n[r] - 2 * new_new_list_g[r]
                    if 0.16 < youxi < 0.22:
                        hetao_T += 1
                        yyxx.append(youxi)
                    else:
                        hetao_F += 1  # 优化1：不符合合套游隙
                    chazhi = math.pow(abs(youxi - 0.19), 2)  # 与最佳游隙的差值
                    fitness_1 += chazhi
                fitness_1 = fitness_1 / len(new_new_list_g)  # 优化2：游隙均值
                fitness_11 = fitness_1 * 100000 + hetao_F
                # print("适应度", fitness_11, "目标：", yyxx, len(yyxx), hetao_T)
            if z == 4 and z == length:
                list_w = []
                list_n = []
                list_g = []
                groud_0_w = []
                groud_1_w = []
                groud_2_w = []
                groud_3_w = []
                groud_0_n = []
                groud_1_n = []
                groud_2_n = []
                groud_3_n = []
                groud_0_g = []
                groud_1_g = []
                groud_2_g = []
                groud_3_g = []
                for t in w:
                    if t < pro_piancha[0]:
                        groud_0_w.append(t)
                    if pro_piancha[0] < t < pro_piancha[1]:
                        groud_1_w.append(t)
                    if pro_piancha[1] < t < pro_piancha[2]:
                        groud_2_w.append(t)
                    if pro_piancha[2] < t:
                        groud_3_w.append(t)
                for u in n:
                    if u < pro_piancha_n[0]:
                        groud_0_n.append(u)
                    if pro_piancha_n[0] < u < pro_piancha_n[1]:
                        groud_1_n.append(u)
                    if pro_piancha_n[1] < u < pro_piancha_n[2]:
                        groud_2_n.append(u)
                    if pro_piancha_n[2] < u:
                        groud_3_n.append(u)
                for h in g:
                    if h < pro_piancha_g[0]:
                        groud_0_g.append(h)
                    if pro_piancha_g[0] < h < pro_piancha_g[1]:
                        groud_1_g.append(h)
                    if pro_piancha_g[1] < h < pro_piancha_g[2]:
                        groud_2_g.append(h)
                    if pro_piancha_g[2] < h:
                        groud_3_g.append(h)
                list_w.append(groud_0_w), list_w.append(groud_1_w), list_w.append(groud_2_w), list_w.append(
                    groud_3_w)
                list_n.append(groud_0_n), list_n.append(groud_1_n), list_n.append(groud_2_n), list_n.append(
                    groud_3_n)
                list_g.append(groud_0_g), list_g.append(groud_1_g), list_g.append(groud_2_g), list_g.append(
                    groud_3_g)
                # 以滚动体为对照排序 即滚动体 ggg = [0,1,2,3,4,5]
                nnn = []  # 内圈
                www = []  # 外圈
                for r in pro_g:
                    T_N = 0
                    for p in pro_n:
                        if p != r:
                            T_N += 1

                        else:
                            nnn.append(T_N)
                    T_W = 0
                    for Z in pro_w:
                        if Z != r:
                            T_W += 1
                        else:
                            www.append(T_W)
                new_list_g = list_g
                new_list_n = []
                new_list_w = []
                for I in nnn:
                    new_list_n.append(list_n[I])
                for H in www:
                    new_list_w.append(list_w[H])
                #####################################################
                length_zu_0 = min(len(new_list_g[0]), len(new_list_n[0]), len(new_list_w[0]))
                length_zu_1 = min(len(new_list_g[1]), len(new_list_n[1]), len(new_list_w[1]))
                length_zu_2 = min(len(new_list_g[2]), len(new_list_n[2]), len(new_list_w[2]))
                length_zu_3 = min(len(new_list_g[3]), len(new_list_n[3]), len(new_list_w[3]))
                length_zu = length_zu_0 + length_zu_1 + length_zu_2 + length_zu_3
                defeat_1_1 = len(w) - length_zu
                new_new_list_g = []
                new_new_list_n = []
                new_new_list_w = []
                new_new_list_g.append(random.sample(new_list_g[0], length_zu_0)), new_new_list_g.append(
                    random.sample(new_list_g[1], length_zu_1)), new_new_list_g.append(
                    random.sample(new_list_g[2], length_zu_2)), new_new_list_g.append(
                    random.sample(new_list_g[3], length_zu_3))
                new_new_list_n.append(random.sample(new_list_n[0], length_zu_0)), new_new_list_n.append(
                    random.sample(new_list_n[1], length_zu_1)), new_new_list_n.append(
                    random.sample(new_list_n[2], length_zu_2)), new_new_list_n.append(
                    random.sample(new_list_n[3], length_zu_3))
                new_new_list_w.append(random.sample(new_list_w[0], length_zu_0)), new_new_list_w.append(
                    random.sample(new_list_w[1], length_zu_1)), new_new_list_w.append(
                    random.sample(new_list_w[2], length_zu_2)), new_new_list_w.append(
                    random.sample(new_list_w[3], length_zu_3))

                new_new_list_g = sum(new_new_list_g, [])  # 列表扁平化处理
                new_new_list_n = sum(new_new_list_n, [])
                new_new_list_w = sum(new_new_list_w, [])
                fitness_1 = 0
                hetao_T = 0
                hetao_F = 0
                yyxx = []
                for r in range(len(new_new_list_g)):  # 优化目标：1.合套率（1.1数量原因没合成 1.2不符合游隙） 2.游隙均值
                    youxi = new_new_list_w[r] - new_new_list_n[r] - 2 * new_new_list_g[r]
                    if 0.16 < youxi < 0.22:
                        hetao_T += 1
                        yyxx.append(youxi)
                    else:
                        hetao_F += 1  # 优化1：不符合合套游隙
                    chazhi = math.pow(abs(youxi - 0.19), 2)  # 与最佳游隙的差值
                    fitness_1 += chazhi
                fitness_1 = fitness_1 / len(new_new_list_g)  # 优化2：游隙均值
                fitness_11 = fitness_1 * 100000 + hetao_F

                # print("适应度", fitness_11, "目标：", yyxx, len(yyxx), hetao_T)
            if z == 5 and z == length:
                list_w = []
                list_n = []
                list_g = []
                groud_0_w = []
                groud_1_w = []
                groud_2_w = []
                groud_3_w = []
                groud_4_w = []
                groud_0_n = []
                groud_1_n = []
                groud_2_n = []
                groud_3_n = []
                groud_4_n = []
                groud_0_g = []
                groud_1_g = []
                groud_2_g = []
                groud_3_g = []
                groud_4_g = []
                for t in w:
                    if t < pro_piancha[0]:
                        groud_0_w.append(t)
                    if pro_piancha[0] < t < pro_piancha[1]:
                        groud_1_w.append(t)
                    if pro_piancha[1] < t < pro_piancha[2]:
                        groud_2_w.append(t)
                    if pro_piancha[2] < t < pro_piancha[3]:
                        groud_3_w.append(t)
                    if pro_piancha[3] < t:
                        groud_4_w.append(t)
                for u in n:
                    if u < pro_piancha_n[0]:
                        groud_0_n.append(u)
                    if pro_piancha_n[0] < u < pro_piancha_n[1]:
                        groud_1_n.append(u)
                    if pro_piancha_n[1] < u < pro_piancha_n[2]:
                        groud_2_n.append(u)
                    if pro_piancha_n[2] < u < pro_piancha_n[3]:
                        groud_3_n.append(u)
                    if pro_piancha_n[3] < u:
                        groud_4_n.append(u)
                for h in g:
                    if h < pro_piancha_g[0]:
                        groud_0_g.append(h)
                    if pro_piancha_g[0] < h < pro_piancha_g[1]:
                        groud_1_g.append(h)
                    if pro_piancha_g[1] < h < pro_piancha_g[2]:
                        groud_2_g.append(h)
                    if pro_piancha_g[2] < h < pro_piancha_g[3]:
                        groud_3_g.append(h)
                    if pro_piancha_g[3] < h:
                        groud_4_g.append(h)
                list_w.append(groud_0_w), list_w.append(groud_1_w), list_w.append(groud_2_w), list_w.append(
                    groud_3_w), list_w.append(groud_4_w)
                list_n.append(groud_0_n), list_n.append(groud_1_n), list_n.append(groud_2_n), list_n.append(
                    groud_3_n), list_n.append(groud_4_n)
                list_g.append(groud_0_g), list_g.append(groud_1_g), list_g.append(groud_2_g), list_g.append(
                    groud_3_g), list_g.append(groud_4_g)
                # 以滚动体为对照排序 即滚动体 ggg = [0,1,2,3,4,5]
                nnn = []  # 内圈
                www = []  # 外圈
                for r in pro_g:
                    T_N = 0
                    for p in pro_n:
                        if p != r:
                            T_N += 1
                        else:
                            nnn.append(T_N)
                    T_W = 0
                    for Z in pro_w:
                        if Z != r:
                            T_W += 1
                        else:
                            www.append(T_W)
                new_list_g = list_g
                new_list_n = []
                new_list_w = []
                for I in nnn:
                    new_list_n.append(list_n[I])
                for H in www:
                    new_list_w.append(list_w[H])
                #####################################################
                length_zu_0 = min(len(new_list_g[0]), len(new_list_n[0]), len(new_list_w[0]))
                length_zu_1 = min(len(new_list_g[1]), len(new_list_n[1]), len(new_list_w[1]))
                length_zu_2 = min(len(new_list_g[2]), len(new_list_n[2]), len(new_list_w[2]))
                length_zu_3 = min(len(new_list_g[3]), len(new_list_n[3]), len(new_list_w[3]))
                length_zu_4 = min(len(new_list_g[4]), len(new_list_n[4]), len(new_list_w[4]))
                length_zu = length_zu_0 + length_zu_1 + length_zu_2 + length_zu_3 + length_zu_4
                defeat_1_1 = len(w) - length_zu
                new_new_list_g = []
                new_new_list_n = []
                new_new_list_w = []
                new_new_list_g.append(random.sample(new_list_g[0], length_zu_0)), new_new_list_g.append(
                    random.sample(new_list_g[1], length_zu_1)), new_new_list_g.append(
                    random.sample(new_list_g[2], length_zu_2)), new_new_list_g.append(
                    random.sample(new_list_g[3], length_zu_3))
                new_new_list_g.append(random.sample(new_list_g[4], length_zu_4))
                new_new_list_n.append(random.sample(new_list_n[0], length_zu_0)), new_new_list_n.append(
                    random.sample(new_list_n[1], length_zu_1)), new_new_list_n.append(
                    random.sample(new_list_n[2], length_zu_2)), new_new_list_n.append(
                    random.sample(new_list_n[3], length_zu_3))
                new_new_list_n.append(random.sample(new_list_n[4], length_zu_4))
                new_new_list_w.append(random.sample(new_list_w[0], length_zu_0)), new_new_list_w.append(
                    random.sample(new_list_w[1], length_zu_1)), new_new_list_w.append(
                    random.sample(new_list_w[2], length_zu_2)), new_new_list_w.append(
                    random.sample(new_list_w[3], length_zu_3))
                new_new_list_w.append(random.sample(new_list_w[4], length_zu_4))

                new_new_list_g = sum(new_new_list_g, [])  # 列表扁平化处理
                new_new_list_n = sum(new_new_list_n, [])
                new_new_list_w = sum(new_new_list_w, [])
                fitness_1 = 0
                hetao_T = 0
                hetao_F = 0
                yyxx = []
                for r in range(len(new_new_list_g)):  # 优化目标：1.合套率（1.1数量原因没合成 1.2不符合游隙） 2.游隙均值
                    youxi = new_new_list_w[r] - new_new_list_n[r] - 2 * new_new_list_g[r]
                    if 0.16 < youxi < 0.22:
                        hetao_T += 1
                        yyxx.append(youxi)
                    else:
                        hetao_F += 1  # 优化1：不符合合套游隙
                    chazhi = math.pow(abs(youxi - 0.19), 2)  # 与最佳游隙的差值
                    fitness_1 += chazhi

                fitness_1 = fitness_1 / len(new_new_list_g)  # 优化2：游隙均值
                fitness_11 = fitness_1 * 100000 + hetao_F
                # print("适应度", fitness_11, "目标：", yyxx, len(yyxx), hetao_T)
            if z == 6 and z == length:
                list_w = []
                list_n = []
                list_g = []
                groud_0_w = []
                groud_1_w = []
                groud_2_w = []
                groud_3_w = []
                groud_4_w = []
                groud_5_w = []
                groud_0_n = []
                groud_1_n = []
                groud_2_n = []
                groud_3_n = []
                groud_4_n = []
                groud_5_n = []
                groud_0_g = []
                groud_1_g = []
                groud_2_g = []
                groud_3_g = []
                groud_4_g = []
                groud_5_g = []
                for t in w:
                    if t < pro_piancha[0]:
                        groud_0_w.append(t)
                    if pro_piancha[0] < t < pro_piancha[1]:
                        groud_1_w.append(t)
                    if pro_piancha[1] < t < pro_piancha[2]:
                        groud_2_w.append(t)
                    if pro_piancha[2] < t < pro_piancha[3]:
                        groud_3_w.append(t)
                    if pro_piancha[3] < t < pro_piancha[4]:
                        groud_4_w.append(t)
                    if pro_piancha[4] < t:
                        groud_5_w.append(t)
                for u in n:
                    if u < pro_piancha_n[0]:
                        groud_0_n.append(u)
                    if pro_piancha_n[0] < u < pro_piancha_n[1]:
                        groud_1_n.append(u)
                    if pro_piancha_n[1] < u < pro_piancha_n[2]:
                        groud_2_n.append(u)
                    if pro_piancha_n[2] < u < pro_piancha_n[3]:
                        groud_3_n.append(u)
                    if pro_piancha_n[3] < u < pro_piancha_n[4]:
                        groud_4_n.append(u)
                    if pro_piancha_n[4] < u:
                        groud_5_n.append(u)
                for h in g:
                    if h < pro_piancha_g[0]:
                        groud_0_g.append(h)
                    if pro_piancha_g[0] < h < pro_piancha_g[1]:
                        groud_1_g.append(h)
                    if pro_piancha_g[1] < h < pro_piancha_g[2]:
                        groud_2_g.append(h)
                    if pro_piancha_g[2] < h < pro_piancha_g[3]:
                        groud_3_g.append(h)
                    if pro_piancha_g[3] < h < pro_piancha_g[4]:
                        groud_4_g.append(h)
                    if pro_piancha_g[4] < h:
                        groud_5_g.append(h)
                list_w.append(groud_0_w), list_w.append(groud_1_w), list_w.append(groud_2_w), list_w.append(
                    groud_3_w), list_w.append(groud_4_w), list_w.append(groud_5_w)
                list_n.append(groud_0_n), list_n.append(groud_1_n), list_n.append(groud_2_n), list_n.append(
                    groud_3_n), list_n.append(groud_4_n), list_n.append(groud_5_n)
                list_g.append(groud_0_g), list_g.append(groud_1_g), list_g.append(groud_2_g), list_g.append(
                    groud_3_g), list_g.append(groud_4_g), list_g.append(groud_5_g)
                # 以滚动体为对照排序 即滚动体 ggg = [0,1,2,3,4,5]
                nnn = []  # 内圈
                www = []  # 外圈
                for r in pro_g:
                    T_N = 0
                    for p in pro_n:
                        if p != r:
                            T_N += 1
                        else:
                            nnn.append(T_N)
                    T_W = 0
                    for Z in pro_w:
                        if Z != r:
                            T_W += 1
                        else:
                            www.append(T_W)
                new_list_g = list_g
                new_list_n = []
                new_list_w = []
                for I in nnn:
                    new_list_n.append(list_n[I])
                for H in www:
                    new_list_w.append(list_w[H])
                #####################################################
                length_zu_0 = min(len(new_list_g[0]), len(new_list_n[0]), len(new_list_w[0]))
                length_zu_1 = min(len(new_list_g[1]), len(new_list_n[1]), len(new_list_w[1]))
                length_zu_2 = min(len(new_list_g[2]), len(new_list_n[2]), len(new_list_w[2]))
                length_zu_3 = min(len(new_list_g[3]), len(new_list_n[3]), len(new_list_w[3]))
                length_zu_4 = min(len(new_list_g[4]), len(new_list_n[4]), len(new_list_w[4]))
                length_zu_5 = min(len(new_list_g[5]), len(new_list_n[5]), len(new_list_w[5]))
                length_zu = length_zu_0 + length_zu_1 + length_zu_2 + length_zu_3 + length_zu_4 + length_zu_5
                defeat_1_1 = len(w) - length_zu
                new_new_list_g = []
                new_new_list_n = []
                new_new_list_w = []
                new_new_list_g.append(random.sample(new_list_g[0], length_zu_0)), new_new_list_g.append(
                    random.sample(new_list_g[1], length_zu_1)), new_new_list_g.append(
                    random.sample(new_list_g[2], length_zu_2)), new_new_list_g.append(
                    random.sample(new_list_g[3], length_zu_3))
                new_new_list_g.append(random.sample(new_list_g[4], length_zu_4)), new_new_list_g.append(
                    random.sample(new_list_g[5], length_zu_5))
                new_new_list_n.append(random.sample(new_list_n[0], length_zu_0)), new_new_list_n.append(
                    random.sample(new_list_n[1], length_zu_1)), new_new_list_n.append(
                    random.sample(new_list_n[2], length_zu_2)), new_new_list_n.append(
                    random.sample(new_list_n[3], length_zu_3))
                new_new_list_n.append(random.sample(new_list_n[4], length_zu_4)), new_new_list_n.append(
                    random.sample(new_list_n[5], length_zu_5))
                new_new_list_w.append(random.sample(new_list_w[0], length_zu_0)), new_new_list_w.append(
                    random.sample(new_list_w[1], length_zu_1)), new_new_list_w.append(
                    random.sample(new_list_w[2], length_zu_2)), new_new_list_w.append(
                    random.sample(new_list_w[3], length_zu_3))
                new_new_list_w.append(random.sample(new_list_w[4], length_zu_4)), new_new_list_w.append(
                    random.sample(new_list_w[5], length_zu_5))

                new_new_list_g = sum(new_new_list_g, [])  # 列表扁平化处理
                new_new_list_n = sum(new_new_list_n, [])
                new_new_list_w = sum(new_new_list_w, [])
                fitness_1 = 0
                hetao_T = 0
                hetao_F = 0
                yyxx = []
                for r in range(len(new_new_list_g)):  # 优化目标：1.合套率（1.1数量原因没合成 1.2不符合游隙） 2.游隙均值
                    youxi = new_new_list_w[r] - new_new_list_n[r] - 2 * new_new_list_g[r]
                    if 0.16 < youxi < 0.22:
                        hetao_T += 1
                        yyxx.append(youxi)
                    else:
                        hetao_F += 1  # 优化1：不符合合套游隙
                    chazhi = math.pow(abs(youxi - 0.19), 2)  # 与最佳游隙的差值
                    fitness_1 += chazhi

                fitness_1 = fitness_1 / len(new_new_list_g)  # 优化2：游隙均值
                fitness_11 = fitness_1 * 100000 + hetao_F
                # print("适应度", fitness_11, "目标：", yyxx, len(yyxx), hetao_T)
            if z == 7 and z == length:
                list_w = []
                list_n = []
                list_g = []
                groud_0_w = []
                groud_1_w = []
                groud_2_w = []
                groud_3_w = []
                groud_4_w = []
                groud_5_w = []
                groud_6_w = []
                groud_0_n = []
                groud_1_n = []
                groud_2_n = []
                groud_3_n = []
                groud_4_n = []
                groud_5_n = []
                groud_6_n = []
                groud_0_g = []
                groud_1_g = []
                groud_2_g = []
                groud_3_g = []
                groud_4_g = []
                groud_5_g = []
                groud_6_g = []
                for t in w:
                    if t < pro_piancha[0]:
                        groud_0_w.append(t)
                    if pro_piancha[0] < t < pro_piancha[1]:
                        groud_1_w.append(t)
                    if pro_piancha[1] < t < pro_piancha[2]:
                        groud_2_w.append(t)
                    if pro_piancha[2] < t < pro_piancha[3]:
                        groud_3_w.append(t)
                    if pro_piancha[3] < t < pro_piancha[4]:
                        groud_4_w.append(t)
                    if pro_piancha[4] < t < pro_piancha[5]:
                        groud_5_w.append(t)
                    if pro_piancha[5] < t:
                        groud_6_w.append(t)
                for u in n:
                    if u < pro_piancha_n[0]:
                        groud_0_n.append(u)
                    if pro_piancha_n[0] < u < pro_piancha_n[1]:
                        groud_1_n.append(u)
                    if pro_piancha_n[1] < u < pro_piancha_n[2]:
                        groud_2_n.append(u)
                    if pro_piancha_n[2] < u < pro_piancha_n[3]:
                        groud_3_n.append(u)
                    if pro_piancha_n[3] < u < pro_piancha_n[4]:
                        groud_4_n.append(u)
                    if pro_piancha_n[4] < u < pro_piancha_n[5]:
                        groud_5_n.append(u)
                    if pro_piancha_n[5] < u:
                        groud_6_n.append(u)
                for h in g:
                    if h < pro_piancha_g[0]:
                        groud_0_g.append(h)
                    if pro_piancha_g[0] < h < pro_piancha_g[1]:
                        groud_1_g.append(h)
                    if pro_piancha_g[1] < h < pro_piancha_g[2]:
                        groud_2_g.append(h)
                    if pro_piancha_g[2] < h < pro_piancha_g[3]:
                        groud_3_g.append(h)
                    if pro_piancha_g[3] < h < pro_piancha_g[4]:
                        groud_4_g.append(h)
                    if pro_piancha_g[4] < h < pro_piancha_g[5]:
                        groud_5_g.append(h)
                    if pro_piancha_g[5] < h:
                        groud_6_g.append(h)
                list_w.append(groud_0_w), list_w.append(groud_1_w), list_w.append(groud_2_w), list_w.append(
                    groud_3_w), list_w.append(groud_4_w), list_w.append(groud_5_w), list_w.append(groud_6_w)
                list_n.append(groud_0_n), list_n.append(groud_1_n), list_n.append(groud_2_n), list_n.append(
                    groud_3_n), list_n.append(groud_4_n), list_n.append(groud_5_n), list_n.append(groud_6_n)
                list_g.append(groud_0_g), list_g.append(groud_1_g), list_g.append(groud_2_g), list_g.append(
                    groud_3_g), list_g.append(groud_4_g), list_g.append(groud_5_g), list_g.append(groud_6_g)
                # 以滚动体为对照排序 即滚动体 ggg = [0,1,2,3,4,5]
                nnn = []  # 内圈
                www = []  # 外圈
                for r in pro_g:
                    T_N = 0
                    for p in pro_n:
                        if p != r:
                            T_N += 1
                        else:
                            nnn.append(T_N)
                    T_W = 0
                    for Z in pro_w:
                        if Z != r:
                            T_W += 1
                        else:
                            www.append(T_W)
                new_list_g = list_g
                new_list_n = []
                new_list_w = []
                for I in nnn:
                    new_list_n.append(list_n[I])
                for H in www:
                    new_list_w.append(list_w[H])
                #####################################################
                length_zu_0 = min(len(new_list_g[0]), len(new_list_n[0]), len(new_list_w[0]))
                length_zu_1 = min(len(new_list_g[1]), len(new_list_n[1]), len(new_list_w[1]))
                length_zu_2 = min(len(new_list_g[2]), len(new_list_n[2]), len(new_list_w[2]))
                length_zu_3 = min(len(new_list_g[3]), len(new_list_n[3]), len(new_list_w[3]))
                length_zu_4 = min(len(new_list_g[4]), len(new_list_n[4]), len(new_list_w[4]))
                length_zu_5 = min(len(new_list_g[5]), len(new_list_n[5]), len(new_list_w[5]))
                length_zu_6 = min(len(new_list_g[6]), len(new_list_n[6]), len(new_list_w[6]))
                length_zu = length_zu_0 + length_zu_1 + length_zu_2 + length_zu_3 + length_zu_4 + length_zu_5 + length_zu_6
                defeat_1_1 = len(w) - length_zu
                new_new_list_g = []
                new_new_list_n = []
                new_new_list_w = []
                new_new_list_g.append(random.sample(new_list_g[0], length_zu_0)), new_new_list_g.append(
                    random.sample(new_list_g[1], length_zu_1)), new_new_list_g.append(
                    random.sample(new_list_g[2], length_zu_2)), new_new_list_g.append(
                    random.sample(new_list_g[3], length_zu_3))
                new_new_list_g.append(random.sample(new_list_g[4], length_zu_4)), new_new_list_g.append(
                    random.sample(new_list_g[5], length_zu_5)), new_new_list_g.append(
                    random.sample(new_list_g[6], length_zu_6))
                new_new_list_n.append(random.sample(new_list_n[0], length_zu_0)), new_new_list_n.append(
                    random.sample(new_list_n[1], length_zu_1)), new_new_list_n.append(
                    random.sample(new_list_n[2], length_zu_2)), new_new_list_n.append(
                    random.sample(new_list_n[3], length_zu_3))
                new_new_list_n.append(random.sample(new_list_n[4], length_zu_4)), new_new_list_n.append(
                    random.sample(new_list_n[5], length_zu_5)), new_new_list_n.append(
                    random.sample(new_list_n[6], length_zu_6))
                new_new_list_w.append(random.sample(new_list_w[0], length_zu_0)), new_new_list_w.append(
                    random.sample(new_list_w[1], length_zu_1)), new_new_list_w.append(
                    random.sample(new_list_w[2], length_zu_2)), new_new_list_w.append(
                    random.sample(new_list_w[3], length_zu_3))
                new_new_list_w.append(random.sample(new_list_w[4], length_zu_4)), new_new_list_w.append(
                    random.sample(new_list_w[5], length_zu_5)), new_new_list_w.append(
                    random.sample(new_list_w[6], length_zu_6))

                new_new_list_g = sum(new_new_list_g, [])  # 列表扁平化处理
                new_new_list_n = sum(new_new_list_n, [])
                new_new_list_w = sum(new_new_list_w, [])
                fitness_1 = 0
                hetao_T = 0
                hetao_F = 0
                yyxx = []
                for r in range(len(new_new_list_g)):  # 优化目标：1.合套率（1.1数量原因没合成 1.2不符合游隙） 2.游隙均值
                    youxi = new_new_list_w[r] - new_new_list_n[r] - 2 * new_new_list_g[r]
                    if 0.16 < youxi < 0.22:
                        hetao_T += 1
                        yyxx.append(youxi)
                    else:
                        hetao_F += 1  # 优化1：不符合合套游隙
                    chazhi = math.pow(abs(youxi - 0.19), 2)  # 与最佳游隙的差值
                    fitness_1 += chazhi

                fitness_1 = fitness_1 / len(new_new_list_g)  # 优化2：游隙均值
                fitness_11 = fitness_1 * 100000 + hetao_F
                # print("适应度", fitness_11, "目标：", yyxx, len(yyxx), hetao_T)
            if z == 8 and z == length:
                list_w = []
                list_n = []
                list_g = []
                groud_0_w = []
                groud_1_w = []
                groud_2_w = []
                groud_3_w = []
                groud_4_w = []
                groud_5_w = []
                groud_6_w = []
                groud_7_w = []
                groud_0_n = []
                groud_1_n = []
                groud_2_n = []
                groud_3_n = []
                groud_4_n = []
                groud_5_n = []
                groud_6_n = []
                groud_7_n = []
                groud_0_g = []
                groud_1_g = []
                groud_2_g = []
                groud_3_g = []
                groud_4_g = []
                groud_5_g = []
                groud_6_g = []
                groud_7_g = []
                for t in w:
                    if t < pro_piancha[0]:
                        groud_0_w.append(t)
                    if pro_piancha[0] < t < pro_piancha[1]:
                        groud_1_w.append(t)
                    if pro_piancha[1] < t < pro_piancha[2]:
                        groud_2_w.append(t)
                    if pro_piancha[2] < t < pro_piancha[3]:
                        groud_3_w.append(t)
                    if pro_piancha[3] < t < pro_piancha[4]:
                        groud_4_w.append(t)
                    if pro_piancha[4] < t < pro_piancha[5]:
                        groud_5_w.append(t)
                    if pro_piancha[5] < t < pro_piancha[6]:
                        groud_6_w.append(t)
                    if pro_piancha[6] < t:
                        groud_7_w.append(t)
                for u in n:
                    if u < pro_piancha_n[0]:
                        groud_0_n.append(u)
                    if pro_piancha_n[0] < u < pro_piancha_n[1]:
                        groud_1_n.append(u)
                    if pro_piancha_n[1] < u < pro_piancha_n[2]:
                        groud_2_n.append(u)
                    if pro_piancha_n[2] < u < pro_piancha_n[3]:
                        groud_3_n.append(u)
                    if pro_piancha_n[3] < u < pro_piancha_n[4]:
                        groud_4_n.append(u)
                    if pro_piancha_n[4] < u < pro_piancha_n[5]:
                        groud_5_n.append(u)
                    if pro_piancha_n[5] < u < pro_piancha_n[6]:
                        groud_6_n.append(u)
                    if pro_piancha_n[6] < u:
                        groud_7_n.append(u)
                for h in g:
                    if h < pro_piancha_g[0]:
                        groud_0_g.append(h)
                    if pro_piancha_g[0] < h < pro_piancha_g[1]:
                        groud_1_g.append(h)
                    if pro_piancha_g[1] < h < pro_piancha_g[2]:
                        groud_2_g.append(h)
                    if pro_piancha_g[2] < h < pro_piancha_g[3]:
                        groud_3_g.append(h)
                    if pro_piancha_g[3] < h < pro_piancha_g[4]:
                        groud_4_g.append(h)
                    if pro_piancha_g[4] < h < pro_piancha_g[5]:
                        groud_5_g.append(h)
                    if pro_piancha_g[5] < h < pro_piancha_g[6]:
                        groud_6_g.append(h)
                    if pro_piancha_g[6] < h:
                        groud_7_g.append(h)
                list_w.append(groud_0_w), list_w.append(groud_1_w), list_w.append(groud_2_w), list_w.append(
                    groud_3_w), list_w.append(groud_4_w), list_w.append(groud_5_w), list_w.append(groud_6_w), list_w.append(
                    groud_7_w)
                list_n.append(groud_0_n), list_n.append(groud_1_n), list_n.append(groud_2_n), list_n.append(
                    groud_3_n), list_n.append(groud_4_n), list_n.append(groud_5_n), list_n.append(groud_6_n), list_n.append(
                    groud_7_n)
                list_g.append(groud_0_g), list_g.append(groud_1_g), list_g.append(groud_2_g), list_g.append(
                    groud_3_g), list_g.append(groud_4_g), list_g.append(groud_5_g), list_g.append(groud_6_g), list_g.append(
                    groud_7_g)
                # 以滚动体为对照排序 即滚动体 ggg = [0,1,2,3,4,5]
                nnn = []  # 内圈
                www = []  # 外圈
                for r in pro_g:
                    T_N = 0
                    for p in pro_n:
                        if p != r:
                            T_N += 1
                        else:
                            nnn.append(T_N)
                    T_W = 0
                    for Z in pro_w:
                        if Z != r:
                            T_W += 1
                        else:
                            www.append(T_W)
                new_list_g = list_g
                new_list_n = []
                new_list_w = []
                for I in nnn:
                    new_list_n.append(list_n[I])
                for H in www:
                    new_list_w.append(list_w[H])
                #####################################################
                length_zu_0 = min(len(new_list_g[0]), len(new_list_n[0]), len(new_list_w[0]))
                length_zu_1 = min(len(new_list_g[1]), len(new_list_n[1]), len(new_list_w[1]))
                length_zu_2 = min(len(new_list_g[2]), len(new_list_n[2]), len(new_list_w[2]))
                length_zu_3 = min(len(new_list_g[3]), len(new_list_n[3]), len(new_list_w[3]))
                length_zu_4 = min(len(new_list_g[4]), len(new_list_n[4]), len(new_list_w[4]))
                length_zu_5 = min(len(new_list_g[5]), len(new_list_n[5]), len(new_list_w[5]))
                length_zu_6 = min(len(new_list_g[6]), len(new_list_n[6]), len(new_list_w[6]))
                length_zu_7 = min(len(new_list_g[7]), len(new_list_n[7]), len(new_list_w[7]))
                length_zu = length_zu_0 + length_zu_1 + length_zu_2 + length_zu_3 + length_zu_4 + length_zu_5 + length_zu_6 + length_zu_7
                defeat_1_1 = len(w) - length_zu
                new_new_list_g = []
                new_new_list_n = []
                new_new_list_w = []
                new_new_list_g.append(random.sample(new_list_g[0], length_zu_0)), new_new_list_g.append(
                    random.sample(new_list_g[1], length_zu_1)), new_new_list_g.append(
                    random.sample(new_list_g[2], length_zu_2)), new_new_list_g.append(
                    random.sample(new_list_g[3], length_zu_3))
                new_new_list_g.append(random.sample(new_list_g[4], length_zu_4)), new_new_list_g.append(
                    random.sample(new_list_g[5], length_zu_5)), new_new_list_g.append(
                    random.sample(new_list_g[6], length_zu_6)), new_new_list_g.append(
                    random.sample(new_list_g[7], length_zu_7))
                new_new_list_n.append(random.sample(new_list_n[0], length_zu_0)), new_new_list_n.append(
                    random.sample(new_list_n[1], length_zu_1)), new_new_list_n.append(
                    random.sample(new_list_n[2], length_zu_2)), new_new_list_n.append(
                    random.sample(new_list_n[3], length_zu_3))
                new_new_list_n.append(random.sample(new_list_n[4], length_zu_4)), new_new_list_n.append(
                    random.sample(new_list_n[5], length_zu_5)), new_new_list_n.append(
                    random.sample(new_list_n[6], length_zu_6)), new_new_list_n.append(
                    random.sample(new_list_n[7], length_zu_7))
                new_new_list_w.append(random.sample(new_list_w[0], length_zu_0)), new_new_list_w.append(
                    random.sample(new_list_w[1], length_zu_1)), new_new_list_w.append(
                    random.sample(new_list_w[2], length_zu_2)), new_new_list_w.append(
                    random.sample(new_list_w[3], length_zu_3))
                new_new_list_w.append(random.sample(new_list_w[4], length_zu_4)), new_new_list_w.append(
                    random.sample(new_list_w[5], length_zu_5)), new_new_list_w.append(
                    random.sample(new_list_w[6], length_zu_6)), new_new_list_w.append(
                    random.sample(new_list_w[7], length_zu_7))

                new_new_list_g = sum(new_new_list_g, [])  # 列表扁平化处理
                new_new_list_n = sum(new_new_list_n, [])
                new_new_list_w = sum(new_new_list_w, [])
                fitness_1 = 0
                hetao_T = 0
                hetao_F = 0
                yyxx = []
                for r in range(len(new_new_list_g)):  # 优化目标：1.合套率（1.1数量原因没合成 1.2不符合游隙） 2.游隙均值
                    youxi = new_new_list_w[r] - new_new_list_n[r] - 2 * new_new_list_g[r]
                    if 0.16 < youxi < 0.22:
                        hetao_T += 1
                        yyxx.append(youxi)
                    else:
                        hetao_F += 1  # 优化1：不符合合套游隙
                    chazhi = math.pow(abs(youxi - 0.19), 2)  # 与最佳游隙的差值
                    fitness_1 += chazhi

                fitness_1 = fitness_1 / len(new_new_list_g)  # 优化2：游隙均值
                fitness_11 = fitness_1 * 100000 + hetao_F
                # print("适应度", fitness_11, "目标：", yyxx, len(yyxx), hetao_T)
            if z == 9 and z == length:
                list_w = []
                list_n = []
                list_g = []
                groud_0_w = []
                groud_1_w = []
                groud_2_w = []
                groud_3_w = []
                groud_4_w = []
                groud_5_w = []
                groud_6_w = []
                groud_7_w = []
                groud_8_w = []
                groud_0_n = []
                groud_1_n = []
                groud_2_n = []
                groud_3_n = []
                groud_4_n = []
                groud_5_n = []
                groud_6_n = []
                groud_7_n = []
                groud_8_n = []
                groud_0_g = []
                groud_1_g = []
                groud_2_g = []
                groud_3_g = []
                groud_4_g = []
                groud_5_g = []
                groud_6_g = []
                groud_7_g = []
                groud_8_g = []
                for t in w:
                    if t < pro_piancha[0]:
                        groud_0_w.append(t)
                    if pro_piancha[0] < t < pro_piancha[1]:
                        groud_1_w.append(t)
                    if pro_piancha[1] < t < pro_piancha[2]:
                        groud_2_w.append(t)
                    if pro_piancha[2] < t < pro_piancha[3]:
                        groud_3_w.append(t)
                    if pro_piancha[3] < t < pro_piancha[4]:
                        groud_4_w.append(t)
                    if pro_piancha[4] < t < pro_piancha[5]:
                        groud_5_w.append(t)
                    if pro_piancha[5] < t < pro_piancha[6]:
                        groud_6_w.append(t)
                    if pro_piancha[6] < t < pro_piancha[7]:
                        groud_7_w.append(t)
                    if pro_piancha[7] < t:
                        groud_8_w.append(t)
                for u in n:
                    if u < pro_piancha_n[0]:
                        groud_0_n.append(u)
                    if pro_piancha_n[0] < u < pro_piancha_n[1]:
                        groud_1_n.append(u)
                    if pro_piancha_n[1] < u < pro_piancha_n[2]:
                        groud_2_n.append(u)
                    if pro_piancha_n[2] < u < pro_piancha_n[3]:
                        groud_3_n.append(u)
                    if pro_piancha_n[3] < u < pro_piancha_n[4]:
                        groud_4_n.append(u)
                    if pro_piancha_n[4] < u < pro_piancha_n[5]:
                        groud_5_n.append(u)
                    if pro_piancha_n[5] < u < pro_piancha_n[6]:
                        groud_6_n.append(u)
                    if pro_piancha_n[6] < u < pro_piancha_n[7]:
                        groud_7_n.append(u)
                    if pro_piancha_n[7] < u:
                        groud_8_n.append(u)
                for h in g:
                    if h < pro_piancha_g[0]:
                        groud_0_g.append(h)
                    if pro_piancha_g[0] < h < pro_piancha_g[1]:
                        groud_1_g.append(h)
                    if pro_piancha_g[1] < h < pro_piancha_g[2]:
                        groud_2_g.append(h)
                    if pro_piancha_g[2] < h < pro_piancha_g[3]:
                        groud_3_g.append(h)
                    if pro_piancha_g[3] < h < pro_piancha_g[4]:
                        groud_4_g.append(h)
                    if pro_piancha_g[4] < h < pro_piancha_g[5]:
                        groud_5_g.append(h)
                    if pro_piancha_g[5] < h < pro_piancha_g[6]:
                        groud_6_g.append(h)
                    if pro_piancha_g[6] < h < pro_piancha_g[7]:
                        groud_7_g.append(h)
                    if pro_piancha_g[7] < h:
                        groud_8_g.append(h)
                list_w.append(groud_0_w), list_w.append(groud_1_w), list_w.append(groud_2_w), list_w.append(
                    groud_3_w), list_w.append(groud_4_w), list_w.append(groud_5_w), list_w.append(
                    groud_6_w), list_w.append(groud_7_w), list_w.append(groud_8_w)
                list_n.append(groud_0_n), list_n.append(groud_1_n), list_n.append(groud_2_n), list_n.append(
                    groud_3_n), list_n.append(groud_4_n), list_n.append(groud_5_n), list_n.append(
                    groud_6_n), list_n.append(groud_7_n), list_n.append(groud_8_n)
                list_g.append(groud_0_g), list_g.append(groud_1_g), list_g.append(groud_2_g), list_g.append(
                    groud_3_g), list_g.append(groud_4_g), list_g.append(groud_5_g), list_g.append(
                    groud_6_g), list_g.append(groud_7_g), list_g.append(groud_8_g)
                # 以滚动体为对照排序 即滚动体 ggg = [0,1,2,3,4,5]
                nnn = []  # 内圈
                www = []  # 外圈
                for r in pro_g:
                    T_N = 0
                    for p in pro_n:
                        if p != r:
                            T_N += 1
                        else:
                            nnn.append(T_N)
                    T_W = 0
                    for Z in pro_w:
                        if Z != r:
                            T_W += 1
                        else:
                            www.append(T_W)
                new_list_g = list_g
                new_list_n = []
                new_list_w = []
                for I in nnn:
                    new_list_n.append(list_n[I])
                for H in www:
                    new_list_w.append(list_w[H])
                #####################################################
                length_zu_0 = min(len(new_list_g[0]), len(new_list_n[0]), len(new_list_w[0]))
                length_zu_1 = min(len(new_list_g[1]), len(new_list_n[1]), len(new_list_w[1]))
                length_zu_2 = min(len(new_list_g[2]), len(new_list_n[2]), len(new_list_w[2]))
                length_zu_3 = min(len(new_list_g[3]), len(new_list_n[3]), len(new_list_w[3]))
                length_zu_4 = min(len(new_list_g[4]), len(new_list_n[4]), len(new_list_w[4]))
                length_zu_5 = min(len(new_list_g[5]), len(new_list_n[5]), len(new_list_w[5]))
                length_zu_6 = min(len(new_list_g[6]), len(new_list_n[6]), len(new_list_w[6]))
                length_zu_7 = min(len(new_list_g[7]), len(new_list_n[7]), len(new_list_w[7]))
                length_zu_8 = min(len(new_list_g[8]), len(new_list_n[8]), len(new_list_w[8]))
                length_zu = length_zu_0 + length_zu_1 + length_zu_2 + length_zu_3 + length_zu_4 + length_zu_5 + length_zu_6 + length_zu_7 + length_zu_8
                defeat_1_1 = len(w) - length_zu
                new_new_list_g = []
                new_new_list_n = []
                new_new_list_w = []
                new_new_list_g.append(random.sample(new_list_g[0], length_zu_0)), new_new_list_g.append(
                    random.sample(new_list_g[1], length_zu_1)), new_new_list_g.append(
                    random.sample(new_list_g[2], length_zu_2)), new_new_list_g.append(
                    random.sample(new_list_g[3], length_zu_3))
                new_new_list_g.append(random.sample(new_list_g[4], length_zu_4)), new_new_list_g.append(
                    random.sample(new_list_g[5], length_zu_5)), new_new_list_g.append(
                    random.sample(new_list_g[6], length_zu_6)), new_new_list_g.append(
                    random.sample(new_list_g[7], length_zu_7)), new_new_list_g.append(
                    random.sample(new_list_g[8], length_zu_8))
                new_new_list_n.append(random.sample(new_list_n[0], length_zu_0)), new_new_list_n.append(
                    random.sample(new_list_n[1], length_zu_1)), new_new_list_n.append(
                    random.sample(new_list_n[2], length_zu_2)), new_new_list_n.append(
                    random.sample(new_list_n[3], length_zu_3))
                new_new_list_n.append(random.sample(new_list_n[4], length_zu_4)), new_new_list_n.append(
                    random.sample(new_list_n[5], length_zu_5)), new_new_list_n.append(
                    random.sample(new_list_n[6], length_zu_6)), new_new_list_n.append(
                    random.sample(new_list_n[7], length_zu_7)), new_new_list_n.append(
                    random.sample(new_list_n[8], length_zu_8))
                new_new_list_w.append(random.sample(new_list_w[0], length_zu_0)), new_new_list_w.append(
                    random.sample(new_list_w[1], length_zu_1)), new_new_list_w.append(
                    random.sample(new_list_w[2], length_zu_2)), new_new_list_w.append(
                    random.sample(new_list_w[3], length_zu_3))
                new_new_list_w.append(random.sample(new_list_w[4], length_zu_4)), new_new_list_w.append(
                    random.sample(new_list_w[5], length_zu_5)), new_new_list_w.append(
                    random.sample(new_list_w[6], length_zu_6)), new_new_list_w.append(
                    random.sample(new_list_w[7], length_zu_7)), new_new_list_w.append(
                    random.sample(new_list_w[8], length_zu_8))

                new_new_list_g = sum(new_new_list_g, [])  # 列表扁平化处理
                new_new_list_n = sum(new_new_list_n, [])
                new_new_list_w = sum(new_new_list_w, [])
                fitness_1 = 0
                hetao_T = 0
                hetao_F = 0
                yyxx = []
                for r in range(len(new_new_list_g)):  # 优化目标：1.合套率（1.1数量原因没合成 1.2不符合游隙） 2.游隙均值
                    youxi = new_new_list_w[r] - new_new_list_n[r] - 2 * new_new_list_g[r]
                    if 0.16 < youxi < 0.22:
                        hetao_T += 1
                        yyxx.append(youxi)
                    else:
                        hetao_F += 1  # 优化1：不符合合套游隙
                    chazhi = math.pow(abs(youxi - 0.19), 2)  # 与最佳游隙的差值
                    fitness_1 += chazhi

                fitness_1 = fitness_1 / len(new_new_list_g)  # 优化2：游隙均值
                fitness_11 = fitness_1 * 100000 + hetao_F
                # print("适应度", fitness_11, "目标：", yyxx, len(yyxx), hetao_T)
            if z == 10 and z == length:
                list_w = []
                list_n = []
                list_g = []
                groud_0_w = []
                groud_1_w = []
                groud_2_w = []
                groud_3_w = []
                groud_4_w = []
                groud_5_w = []
                groud_6_w = []
                groud_7_w = []
                groud_8_w = []
                groud_9_w = []
                groud_0_n = []
                groud_1_n = []
                groud_2_n = []
                groud_3_n = []
                groud_4_n = []
                groud_5_n = []
                groud_6_n = []
                groud_7_n = []
                groud_8_n = []
                groud_9_n = []
                groud_0_g = []
                groud_1_g = []
                groud_2_g = []
                groud_3_g = []
                groud_4_g = []
                groud_5_g = []
                groud_6_g = []
                groud_7_g = []
                groud_8_g = []
                groud_9_g = []
                for t in w:
                    if t < pro_piancha[0]:
                        groud_0_w.append(t)
                    if pro_piancha[0] < t < pro_piancha[1]:
                        groud_1_w.append(t)
                    if pro_piancha[1] < t < pro_piancha[2]:
                        groud_2_w.append(t)
                    if pro_piancha[2] < t < pro_piancha[3]:
                        groud_3_w.append(t)
                    if pro_piancha[3] < t < pro_piancha[4]:
                        groud_4_w.append(t)
                    if pro_piancha[4] < t < pro_piancha[5]:
                        groud_5_w.append(t)
                    if pro_piancha[5] < t < pro_piancha[6]:
                        groud_6_w.append(t)
                    if pro_piancha[6] < t < pro_piancha[7]:
                        groud_7_w.append(t)
                    if pro_piancha[7] < t < pro_piancha[8]:
                        groud_8_w.append(t)
                    if pro_piancha[8] < t:
                        groud_9_w.append(t)
                for u in n:
                    if u < pro_piancha_n[0]:
                        groud_0_n.append(u)
                    if pro_piancha_n[0] < u < pro_piancha_n[1]:
                        groud_1_n.append(u)
                    if pro_piancha_n[1] < u < pro_piancha_n[2]:
                        groud_2_n.append(u)
                    if pro_piancha_n[2] < u < pro_piancha_n[3]:
                        groud_3_n.append(u)
                    if pro_piancha_n[3] < u < pro_piancha_n[4]:
                        groud_4_n.append(u)
                    if pro_piancha_n[4] < u < pro_piancha_n[5]:
                        groud_5_n.append(u)
                    if pro_piancha_n[5] < u < pro_piancha_n[6]:
                        groud_6_n.append(u)
                    if pro_piancha_n[6] < u < pro_piancha_n[7]:
                        groud_7_n.append(u)
                    if pro_piancha_n[7] < u < pro_piancha_n[8]:
                        groud_8_n.append(u)
                    if pro_piancha_n[8] < u:
                        groud_9_n.append(u)
                for h in g:
                    if h < pro_piancha_g[0]:
                        groud_0_g.append(h)
                    if pro_piancha_g[0] < h < pro_piancha_g[1]:
                        groud_1_g.append(h)
                    if pro_piancha_g[1] < h < pro_piancha_g[2]:
                        groud_2_g.append(h)
                    if pro_piancha_g[2] < h < pro_piancha_g[3]:
                        groud_3_g.append(h)
                    if pro_piancha_g[3] < h < pro_piancha_g[4]:
                        groud_4_g.append(h)
                    if pro_piancha_g[4] < h < pro_piancha_g[5]:
                        groud_5_g.append(h)
                    if pro_piancha_g[5] < h < pro_piancha_g[6]:
                        groud_6_g.append(h)
                    if pro_piancha_g[6] < h < pro_piancha_g[7]:
                        groud_7_g.append(h)
                    if pro_piancha_g[7] < h < pro_piancha_g[8]:
                        groud_8_g.append(h)
                    if pro_piancha_g[8] < h:
                        groud_9_g.append(h)
                list_w.append(groud_0_w), list_w.append(groud_1_w), list_w.append(groud_2_w), list_w.append(
                    groud_3_w), list_w.append(groud_4_w), list_w.append(groud_5_w), list_w.append(
                    groud_6_w), list_w.append(groud_7_w), list_w.append(groud_8_w), list_w.append(groud_9_w)
                list_n.append(groud_0_n), list_n.append(groud_1_n), list_n.append(groud_2_n), list_n.append(
                    groud_3_n), list_n.append(groud_4_n), list_n.append(groud_5_n), list_n.append(
                    groud_6_n), list_n.append(groud_7_n), list_n.append(groud_8_n), list_n.append(groud_9_n)
                list_g.append(groud_0_g), list_g.append(groud_1_g), list_g.append(groud_2_g), list_g.append(
                    groud_3_g), list_g.append(groud_4_g), list_g.append(groud_5_g), list_g.append(
                    groud_6_g), list_g.append(groud_7_g), list_g.append(groud_8_g), list_g.append(groud_9_g)
                # 以滚动体为对照排序 即滚动体 ggg = [0,1,2,3,4,5]
                nnn = []  # 内圈
                www = []  # 外圈
                for r in pro_g:
                    T_N = 0
                    for p in pro_n:
                        if p != r:
                            T_N += 1
                        else:
                            nnn.append(T_N)
                    T_W = 0
                    for Z in pro_w:
                        if Z != r:
                            T_W += 1
                        else:
                            www.append(T_W)
                new_list_g = list_g
                new_list_n = []
                new_list_w = []
                for I in nnn:
                    new_list_n.append(list_n[I])
                for H in www:
                    new_list_w.append(list_w[H])
                #####################################################
                length_zu_0 = min(len(new_list_g[0]), len(new_list_n[0]), len(new_list_w[0]))
                length_zu_1 = min(len(new_list_g[1]), len(new_list_n[1]), len(new_list_w[1]))
                length_zu_2 = min(len(new_list_g[2]), len(new_list_n[2]), len(new_list_w[2]))
                length_zu_3 = min(len(new_list_g[3]), len(new_list_n[3]), len(new_list_w[3]))
                length_zu_4 = min(len(new_list_g[4]), len(new_list_n[4]), len(new_list_w[4]))
                length_zu_5 = min(len(new_list_g[5]), len(new_list_n[5]), len(new_list_w[5]))
                length_zu_6 = min(len(new_list_g[6]), len(new_list_n[6]), len(new_list_w[6]))
                length_zu_7 = min(len(new_list_g[7]), len(new_list_n[7]), len(new_list_w[7]))
                length_zu_8 = min(len(new_list_g[8]), len(new_list_n[8]), len(new_list_w[8]))
                length_zu_9 = min(len(new_list_g[9]), len(new_list_n[9]), len(new_list_w[9]))
                length_zu = length_zu_0 + length_zu_1 + length_zu_2 + length_zu_3 + length_zu_4 + length_zu_5 + length_zu_6 + length_zu_7 + length_zu_8 + length_zu_9
                defeat_1_1 = len(w) - length_zu
                new_new_list_g = []
                new_new_list_n = []
                new_new_list_w = []
                new_new_list_g.append(random.sample(new_list_g[0], length_zu_0)), new_new_list_g.append(
                    random.sample(new_list_g[1], length_zu_1)), new_new_list_g.append(
                    random.sample(new_list_g[2], length_zu_2)), new_new_list_g.append(
                    random.sample(new_list_g[3], length_zu_3))
                new_new_list_g.append(random.sample(new_list_g[4], length_zu_4)), new_new_list_g.append(
                    random.sample(new_list_g[5], length_zu_5)), new_new_list_g.append(
                    random.sample(new_list_g[6], length_zu_6)), new_new_list_g.append(
                    random.sample(new_list_g[7], length_zu_7)), new_new_list_g.append(
                    random.sample(new_list_g[8], length_zu_8)), new_new_list_g.append(
                    random.sample(new_list_g[9], length_zu_9))
                new_new_list_n.append(random.sample(new_list_n[0], length_zu_0)), new_new_list_n.append(
                    random.sample(new_list_n[1], length_zu_1)), new_new_list_n.append(
                    random.sample(new_list_n[2], length_zu_2)), new_new_list_n.append(
                    random.sample(new_list_n[3], length_zu_3))
                new_new_list_n.append(random.sample(new_list_n[4], length_zu_4)), new_new_list_n.append(
                    random.sample(new_list_n[5], length_zu_5)), new_new_list_n.append(
                    random.sample(new_list_n[6], length_zu_6)), new_new_list_n.append(
                    random.sample(new_list_n[7], length_zu_7)), new_new_list_n.append(
                    random.sample(new_list_n[8], length_zu_8)), new_new_list_n.append(
                    random.sample(new_list_n[9], length_zu_9))
                new_new_list_w.append(random.sample(new_list_w[0], length_zu_0)), new_new_list_w.append(
                    random.sample(new_list_w[1], length_zu_1)), new_new_list_w.append(
                    random.sample(new_list_w[2], length_zu_2)), new_new_list_w.append(
                    random.sample(new_list_w[3], length_zu_3))
                new_new_list_w.append(random.sample(new_list_w[4], length_zu_4)), new_new_list_w.append(
                    random.sample(new_list_w[5], length_zu_5)), new_new_list_w.append(
                    random.sample(new_list_w[6], length_zu_6)), new_new_list_w.append(
                    random.sample(new_list_w[7], length_zu_7)), new_new_list_w.append(
                    random.sample(new_list_w[8], length_zu_8)), new_new_list_w.append(
                    random.sample(new_list_w[9], length_zu_9))

                new_new_list_g = sum(new_new_list_g, [])  # 列表扁平化处理
                new_new_list_n = sum(new_new_list_n, [])
                new_new_list_w = sum(new_new_list_w, [])
                fitness_1 = 0
                hetao_T = 0
                hetao_F = 0
                yyxx = []
                for r in range(len(new_new_list_g)):  # 优化目标：1.合套率（1.1数量原因没合成 1.2不符合游隙） 2.游隙均值
                    youxi = new_new_list_w[r] - new_new_list_n[r] - 2 * new_new_list_g[r]
                    if 0.16 < youxi < 0.22:
                        hetao_T += 1
                        yyxx.append(youxi)
                    else:
                        hetao_F += 1  # 优化1：不符合合套游隙
                    chazhi = math.pow(abs(youxi - 0.19), 2)  # 与最佳游隙的差值
                    fitness_1 += chazhi

                fitness_1 = fitness_1 / len(new_new_list_g)  # 优化2：游隙均值
                fitness_11 = fitness_1 * 100000 + hetao_F
                # print("适应度", fitness_11, "目标：", yyxx, len(yyxx), hetao_T)
        time4 = datetime.datetime.now()
        # print("概率划分运行时间_3：", (time4 - time3).seconds)
        return fitness_11, pro


    def destroy_operator(self, solution, num_destroy):
        # 破坏算子: 随机破坏
        a_list = [i for i in range(10)]
        k_1 = random.sample(a_list, int(num_destroy/3))
        k_2 = random.sample(a_list, int(num_destroy/3))
        k_3 = random.sample(a_list, int(num_destroy/3))
        destroy_node_bank = k_1+k_2+k_3

        route_1 = solution.route[:10]
        route_2 = solution.route[10:20]
        route_3 = solution.route[20:30]
        for h in range(int(len(k_1))):
            route_1.remove(k_1[h])
            route_2.remove(k_2[h])
            route_3.remove(k_3[h])
        solution.route = route_1+route_2+route_3
        return solution, destroy_node_bank

    def repair_operator(self, solution, destroy_node_bank):
        # 修复算子: 随机修复
        s_route_1 = solution.route[:7]
        for w in range(3):
            in_dex_1 = random.sample([e for e in range(8+w)], 1)
            s_route_1.insert(in_dex_1[0],destroy_node_bank[:3][w])

        s_route_2 = solution.route[7:14]
        for n in range(3):
            in_dex_2 = random.sample([e for e in range(8+n)], 1)
            s_route_2.insert(in_dex_2[0], destroy_node_bank[3:6][n])

        s_route_3 = solution.route[14:]
        for g in range(3):
            in_dex_3 = random.sample([e for e in range(8+g)], 1)
            s_route_3.insert(in_dex_3[0], destroy_node_bank[6:9][g])

        solution.route = s_route_1 + s_route_2 + s_route_3
        return solution

def plot_best_vales_iteration(best_values_record):
    # 绘制最优解随着迭代变化的趋势
    plt.figure()
    plt.plot([i+1 for i in range(len(best_values_record))], best_values_record)
    plt.xlabel('迭代次数')
    plt.ylabel('最优值')
    plt.show()


if __name__ == '__main__':
    ############## 算例和参数设置 ############################
    time_0 = datetime.datetime.now()
    # input
    np.random.seed(0)
    w = np.random.normal(loc=308.92, scale=0.00500, size=1000)
    n = np.random.normal(loc=244.745, scale=0.01333, size=1000)
    g = np.random.normal(loc=31.9925, scale=0.00417, size=1000)

    iter_num = 2000  # 迭代次数
    num_destroy = 9 # 破坏程度

    ############## 产生初始解 ############################
    solution = Solution()
    solution.route = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    LT = Lns_tsp()
    solution.cost, solution.pro= LT.get_route_cost(solution.route) # 计算初始解对应的目标成本
    best_solution = deepcopy(solution)  # 初始化最优解=初始解
    best_values_record = [0 for i in range(iter_num)]  # 初始化保存最优解的集合

    ############## 执行LNS ############################
    for n_gen in range(iter_num):
        tem_solution = deepcopy(solution)
        # 执行破坏修复算子，得到临时解
        tem_solution, destroy_node_bank = LT.destroy_operator(tem_solution, num_destroy)
        tem_solution = LT.repair_operator(tem_solution, destroy_node_bank)
        # 计算临时解的目标值
        tem_solution.cost,tem_solution.pro = LT.get_route_cost(tem_solution.route)

        # 接受标准：如果临时解比当前解好，直接接受；且更新最优解
        if tem_solution.cost < best_solution.cost:
            solution = deepcopy(tem_solution)
            best_solution = deepcopy(tem_solution)
        best_values_record[n_gen] = best_solution.cost

    ############## 绘制结果 ############################

    print(best_solution.route)
    print(best_solution.cost)
    print(best_solution.pro)
    time_1 = datetime.datetime.now()
    print("LNS划分运行时间：", (time_1 - time_0).seconds)
    plt.plot([i for i in range(int(len(best_values_record)))], best_values_record, 'r-')
    plt.ylabel("游隙指标")
    plt.xlabel("迭代次数")
    plt.legend(["LNS"])
    plt.show()
# 结果  迭代250000次，速度较快，可以考虑更大的迭代次数，找寻更好的解
# [9, 4, 0, 5, 2, 8, 6, 3, 1, 7, 8, 4, 0, 7, 5, 2, 3, 6, 9, 1, 1, 7, 3, 8, 2, 6, 9, 5, 0, 4]
# 1.742851793413067
# [0.25817592 0.03940069 0.19257226 0.15500184 0.06847987 0.28636942]