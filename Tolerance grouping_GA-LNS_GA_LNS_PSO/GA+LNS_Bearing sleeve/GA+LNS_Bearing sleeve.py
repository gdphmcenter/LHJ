import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import math
plt.rcParams['font.sans-serif'] = ['SimHei']
from copy import deepcopy
import scipy.stats as st
import datetime
np.random.seed(0)                                                       # 滚动体尺寸偏差 N( u = 10.75, * = 2.75 )
w = np.random.normal(loc=308.92,scale=0.00500,size=1000)

n = np.random.normal(loc=244.745,scale=0.01333,size=1000)

g = np.random.normal(loc=31.9925,scale=0.00417,size=1000)

geti_Norm = np.concatenate((w,n,g),axis=0)

class Individual:
    def __init__(self,genes = None ):
        # 随机生成序
        if genes == None:
            genes1 = np.random.randint(0,2,10)
            # genes1 = [1,0,1,1,0,0,0,0,0,0]
            genes1 = genes1.tolist()
            genes2 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
            genes3 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
            genes4 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
            random.shuffle(genes2)
            random.shuffle(genes3)
            random.shuffle(genes4)
            genes = genes1+genes2+genes3+genes4
        self.genes = genes                                 # gense = [0,1,2,3,4,5,6...29]
        self.fitness,self.pro = self.evaluate_fitness()
    def evaluate_fitness(self):
        # 计算个体适应度
        fitness = []
        time1 = datetime.datetime.now()
        num_zu = 0       # 组数
        zero_hot = self.genes[:10]  # 0/1编码确定组数，随机概率
        w_zu = self.genes[10:20]    # 外圈组
        n_zu = self.genes[20:30]    # 内圈组
        g_zu = self.genes[30:40]    # 滚动体组
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
                pro_0 = np.random.dirichlet(np.ones(num_zu), size=1)      # 随机概率
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
            if h <  num_zu:
                t_w.append(h)
            else:
                pass
        for p in n_zu:
            if p <  num_zu:
                t_n.append(p)
            else:
                pass
        for j in g_zu:
            if j <  num_zu:
                t_g.append(j)
            else:
                pass
            # 外圈、内圈、滚动体概率
        pro_w = []    # 已知组的个数，外圈的组概率
        pro_n = []    # 已知组的个数，内圈的组概率
        pro_g = []    # 已知组的个数，滚动体的组概率
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
        strattime = datetime.datetime.now() #开始时间

        if len(pro_w) > 1:
            for P in range(1,len(pro_w)):
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
        for z in range(11):    # 0-10 闭区间                  # 统一返回一个变量，再在最后用return
            if z == 0 and z == length:          # 0组
                fitness_11 = 999999999 # 可以给一个很大的值，来淘汰掉
                yyxx = []
            if z == 1 and z == length :         # 1组  全部合套，没有被排除的，只考虑合套游隙  一组的合套效果都不咋好
                fitness_1 = 0
                hetao_T = 0
                hetao_F = 0
                groud_0_w = w
                groud_0_n = n
                groud_0_g = g
                yyxx = []
                for r in range(len(w)):           # 优化目标：1.合套率（1.1数量原因没合成 1.2不符合游隙） 2.游隙均值  3.数据的方差
                    youxi = groud_0_w[r] - groud_0_n[r] - 2*groud_0_g[r]
                    if  0.16 < youxi < 0.22:
                        hetao_T += 1
                        yyxx.append(youxi)
                    else:
                        hetao_F +=1                         # 优化1：不符合合套游隙
                    chazhi = math.pow(abs(youxi-0.19),2)       # 与最佳游隙的差值
                    fitness_1 += chazhi
                fitness_1 = fitness_1/len(w)                           # 优化2：游隙均值
                fitness_11 = fitness_1*100000 + hetao_F
                # print("适应度", fitness_11, "目标：", yyxx, len(yyxx), hetao_T)
            if z == 2 and z == length:          # 2组          # 优化目标：1.合套率（1.1数量原因没合成 1.2不符合游隙） 2.游隙均值
                list_w = []
                list_n = []
                list_g = []
                groud_0_w = []     # 4
                groud_1_w = []     # 23
                groud_0_n = []     # 20
                groud_1_n = []     # 6
                groud_0_g = []     # 5
                groud_1_g = []     # 21
                for t in w:
                    if t < pro_piancha[0]:
                        groud_0_w.append(t)
                    if pro_piancha[0] < t :
                        groud_1_w.append(t)

                for u in n:
                    if u < pro_piancha_n[0]:
                        groud_0_n.append(u)
                    if pro_piancha_n[0] < u :
                        groud_1_n.append(u)
                for h in g:
                    if h < pro_piancha_g[0]:
                        groud_0_g.append(h)
                    if pro_piancha_g[0] < h :
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
                new_new_list_g.append(random.sample(new_list_g[0],length_zu_0)), new_new_list_g.append(
                    random.sample(new_list_g[1],length_zu_1))
                new_new_list_n.append(random.sample(new_list_n[0],length_zu_0)), new_new_list_n.append(
                    random.sample(new_list_n[1],length_zu_1))
                new_new_list_w.append(random.sample(new_list_w[0],length_zu_0)), new_new_list_w.append(
                    random.sample(new_list_w[1],length_zu_1))
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
                    chazhi = math.pow(abs(youxi - 0.19),2)  # 与最佳游隙的差值
                    fitness_1 += chazhi


                fitness_1 = fitness_1 / len(new_new_list_g)  # 优化2：游隙均值
                fitness_11 = fitness_1*100000 + hetao_F
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
                new_new_list_g.append(random.sample(new_list_g[0],length_zu_0)), new_new_list_g.append(
                    random.sample(new_list_g[1],length_zu_1)), new_new_list_g.append(
                    random.sample(new_list_g[2],length_zu_2))
                new_new_list_n.append(random.sample(new_list_n[0],length_zu_0)), new_new_list_n.append(
                    random.sample(new_list_n[1],length_zu_1)), new_new_list_n.append(
                    random.sample(new_list_n[2],length_zu_2))
                new_new_list_w.append(random.sample(new_list_w[0],length_zu_0)), new_new_list_w.append(
                    random.sample(new_list_w[1],length_zu_1)), new_new_list_w.append(
                    random.sample(new_list_w[2],length_zu_2))


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
                    chazhi = math.pow(abs(youxi - 0.19),2)  # 与最佳游隙的差值
                    fitness_1 += chazhi
                fitness_1 = fitness_1 / len(new_new_list_g)  # 优化2：游隙均值
                fitness_11 = fitness_1*100000 + hetao_F
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
                new_new_list_g.append(random.sample(new_list_g[0],length_zu_0)), new_new_list_g.append(
                    random.sample(new_list_g[1],length_zu_1)), new_new_list_g.append(
                    random.sample(new_list_g[2],length_zu_2)), new_new_list_g.append(random.sample(new_list_g[3],length_zu_3))
                new_new_list_n.append(random.sample(new_list_n[0],length_zu_0)), new_new_list_n.append(
                    random.sample(new_list_n[1],length_zu_1)), new_new_list_n.append(
                    random.sample(new_list_n[2],length_zu_2)), new_new_list_n.append(random.sample(new_list_n[3],length_zu_3))
                new_new_list_w.append(random.sample(new_list_w[0],length_zu_0)), new_new_list_w.append(
                    random.sample(new_list_w[1],length_zu_1)), new_new_list_w.append(
                    random.sample(new_list_w[2],length_zu_2)), new_new_list_w.append(random.sample(new_list_w[3],length_zu_3))


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
                    chazhi = math.pow(abs(youxi - 0.19),2)  # 与最佳游隙的差值
                    fitness_1 += chazhi
                fitness_1 = fitness_1 / len(new_new_list_g)  # 优化2：游隙均值
                fitness_11 = fitness_1*100000 + hetao_F

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
                    if pro_piancha[3] < t :
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
                new_new_list_g.append(random.sample(new_list_g[0],length_zu_0)), new_new_list_g.append(
                    random.sample(new_list_g[1],length_zu_1)), new_new_list_g.append(
                    random.sample(new_list_g[2],length_zu_2)), new_new_list_g.append(random.sample(new_list_g[3],length_zu_3))
                new_new_list_g.append(random.sample(new_list_g[4],length_zu_4))
                new_new_list_n.append(random.sample(new_list_n[0],length_zu_0)), new_new_list_n.append(
                    random.sample(new_list_n[1],length_zu_1)), new_new_list_n.append(
                    random.sample(new_list_n[2],length_zu_2)), new_new_list_n.append(random.sample(new_list_n[3],length_zu_3))
                new_new_list_n.append(random.sample(new_list_n[4],length_zu_4))
                new_new_list_w.append(random.sample(new_list_w[0],length_zu_0)), new_new_list_w.append(
                    random.sample(new_list_w[1],length_zu_1)), new_new_list_w.append(
                    random.sample(new_list_w[2],length_zu_2)), new_new_list_w.append(random.sample(new_list_w[3],length_zu_3))
                new_new_list_w.append(random.sample(new_list_w[4],length_zu_4))



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
                    chazhi = math.pow(abs(youxi - 0.19),2)  # 与最佳游隙的差值
                    fitness_1 += chazhi

                fitness_1 = fitness_1 / len(new_new_list_g)  # 优化2：游隙均值
                fitness_11 = fitness_1*100000 + hetao_F
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
                    if pro_piancha[4] < t :
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
                list_w.append(groud_0_w), list_w.append(groud_1_w), list_w.append(groud_2_w), list_w.append(groud_3_w), list_w.append(groud_4_w), list_w.append(groud_5_w)
                list_n.append(groud_0_n), list_n.append(groud_1_n), list_n.append(groud_2_n), list_n.append(groud_3_n), list_n.append(groud_4_n), list_n.append(groud_5_n)
                list_g.append(groud_0_g), list_g.append(groud_1_g), list_g.append(groud_2_g), list_g.append(groud_3_g), list_g.append(groud_4_g), list_g.append(groud_5_g)
                # 以滚动体为对照排序 即滚动体 ggg = [0,1,2,3,4,5]
                nnn = []    # 内圈
                www = []    # 外圈
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
                length_zu = length_zu_0+length_zu_1+length_zu_2+length_zu_3+length_zu_4+length_zu_5
                defeat_1_1 = len(w)-length_zu
                new_new_list_g = []
                new_new_list_n = []
                new_new_list_w = []
                new_new_list_g.append(random.sample(new_list_g[0],length_zu_0)), new_new_list_g.append(random.sample(new_list_g[1],length_zu_1)), new_new_list_g.append(random.sample(new_list_g[2],length_zu_2)), new_new_list_g.append(random.sample(new_list_g[3],length_zu_3))
                new_new_list_g.append(random.sample(new_list_g[4],length_zu_4)), new_new_list_g.append(random.sample(new_list_g[5],length_zu_5))
                new_new_list_n.append(random.sample(new_list_n[0],length_zu_0)), new_new_list_n.append(random.sample(new_list_n[1],length_zu_1)), new_new_list_n.append(random.sample(new_list_n[2],length_zu_2)), new_new_list_n.append(random.sample(new_list_n[3],length_zu_3))
                new_new_list_n.append(random.sample(new_list_n[4],length_zu_4)), new_new_list_n.append(random.sample(new_list_n[5],length_zu_5))
                new_new_list_w.append(random.sample(new_list_w[0],length_zu_0)), new_new_list_w.append(random.sample(new_list_w[1],length_zu_1)), new_new_list_w.append(random.sample(new_list_w[2],length_zu_2)), new_new_list_w.append(random.sample(new_list_w[3],length_zu_3))
                new_new_list_w.append(random.sample(new_list_w[4],length_zu_4)),new_new_list_w.append(random.sample(new_list_w[5],length_zu_5))



                new_new_list_g = sum(new_new_list_g,[])     # 列表扁平化处理
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
                    chazhi = math.pow(abs(youxi - 0.19),2)  # 与最佳游隙的差值
                    fitness_1 += chazhi

                fitness_1 = fitness_1 / len(new_new_list_g)  # 优化2：游隙均值
                fitness_11 = fitness_1*100000 + hetao_F
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
                    if pro_piancha[5] < t :
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
                new_new_list_g.append(random.sample(new_list_g[0],length_zu_0)), new_new_list_g.append(
                    random.sample(new_list_g[1],length_zu_1)), new_new_list_g.append(
                    random.sample(new_list_g[2],length_zu_2)), new_new_list_g.append(random.sample(new_list_g[3],length_zu_3))
                new_new_list_g.append(random.sample(new_list_g[4],length_zu_4)), new_new_list_g.append(random.sample(new_list_g[5],length_zu_5)), new_new_list_g.append(random.sample(new_list_g[6],length_zu_6))
                new_new_list_n.append(random.sample(new_list_n[0],length_zu_0)), new_new_list_n.append(
                    random.sample(new_list_n[1],length_zu_1)), new_new_list_n.append(
                    random.sample(new_list_n[2],length_zu_2)), new_new_list_n.append(random.sample(new_list_n[3],length_zu_3))
                new_new_list_n.append(random.sample(new_list_n[4],length_zu_4)), new_new_list_n.append(random.sample(new_list_n[5],length_zu_5)), new_new_list_n.append(random.sample(new_list_n[6],length_zu_6))
                new_new_list_w.append(random.sample(new_list_w[0],length_zu_0)), new_new_list_w.append(
                    random.sample(new_list_w[1],length_zu_1)), new_new_list_w.append(
                    random.sample(new_list_w[2],length_zu_2)), new_new_list_w.append(random.sample(new_list_w[3],length_zu_3))
                new_new_list_w.append(random.sample(new_list_w[4],length_zu_4)), new_new_list_w.append(random.sample(new_list_w[5],length_zu_5)), new_new_list_w.append(random.sample(new_list_w[6],length_zu_6))



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
                    chazhi = math.pow(abs(youxi - 0.19),2)  # 与最佳游隙的差值
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
                    if pro_piancha[6] < t :
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
                    groud_3_w), list_w.append(groud_4_w), list_w.append(groud_5_w), list_w.append(groud_6_w), list_w.append(groud_7_w)
                list_n.append(groud_0_n), list_n.append(groud_1_n), list_n.append(groud_2_n), list_n.append(
                    groud_3_n), list_n.append(groud_4_n), list_n.append(groud_5_n), list_n.append(groud_6_n), list_n.append(groud_7_n)
                list_g.append(groud_0_g), list_g.append(groud_1_g), list_g.append(groud_2_g), list_g.append(
                    groud_3_g), list_g.append(groud_4_g), list_g.append(groud_5_g), list_g.append(groud_6_g), list_g.append(groud_7_g)
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
                new_new_list_g.append(random.sample(new_list_g[0],length_zu_0)), new_new_list_g.append(
                    random.sample(new_list_g[1],length_zu_1)), new_new_list_g.append(
                    random.sample(new_list_g[2],length_zu_2)), new_new_list_g.append(random.sample(new_list_g[3],length_zu_3))
                new_new_list_g.append(random.sample(new_list_g[4],length_zu_4)), new_new_list_g.append(
                    random.sample(new_list_g[5],length_zu_5)), new_new_list_g.append(random.sample(new_list_g[6],length_zu_6)), new_new_list_g.append(random.sample(new_list_g[7],length_zu_7))
                new_new_list_n.append(random.sample(new_list_n[0],length_zu_0)), new_new_list_n.append(
                    random.sample(new_list_n[1],length_zu_1)), new_new_list_n.append(
                    random.sample(new_list_n[2],length_zu_2)), new_new_list_n.append(random.sample(new_list_n[3],length_zu_3))
                new_new_list_n.append(random.sample(new_list_n[4],length_zu_4)), new_new_list_n.append(
                    random.sample(new_list_n[5],length_zu_5)), new_new_list_n.append(random.sample(new_list_n[6],length_zu_6)), new_new_list_n.append(random.sample(new_list_n[7],length_zu_7))
                new_new_list_w.append(random.sample(new_list_w[0],length_zu_0)), new_new_list_w.append(
                    random.sample(new_list_w[1],length_zu_1)), new_new_list_w.append(
                    random.sample(new_list_w[2],length_zu_2)), new_new_list_w.append(random.sample(new_list_w[3],length_zu_3))
                new_new_list_w.append(random.sample(new_list_w[4],length_zu_4)), new_new_list_w.append(
                    random.sample(new_list_w[5],length_zu_5)), new_new_list_w.append(random.sample(new_list_w[6],length_zu_6)), new_new_list_w.append(random.sample(new_list_w[7],length_zu_7))


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
                    chazhi = math.pow(abs(youxi - 0.19),2)  # 与最佳游隙的差值
                    fitness_1 += chazhi

                fitness_1 = fitness_1 / len(new_new_list_g)  # 优化2：游隙均值
                fitness_11 = fitness_1*100000 + hetao_F
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
                    if pro_piancha[7] < t :
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
                new_new_list_g.append(random.sample(new_list_g[0],length_zu_0)), new_new_list_g.append(
                    random.sample(new_list_g[1],length_zu_1)), new_new_list_g.append(
                    random.sample(new_list_g[2],length_zu_2)), new_new_list_g.append(random.sample(new_list_g[3],length_zu_3))
                new_new_list_g.append(random.sample(new_list_g[4],length_zu_4)), new_new_list_g.append(
                    random.sample(new_list_g[5],length_zu_5)), new_new_list_g.append(
                    random.sample(new_list_g[6],length_zu_6)), new_new_list_g.append(random.sample(new_list_g[7],length_zu_7)), new_new_list_g.append(random.sample(new_list_g[8],length_zu_8))
                new_new_list_n.append(random.sample(new_list_n[0],length_zu_0)), new_new_list_n.append(
                    random.sample(new_list_n[1],length_zu_1)), new_new_list_n.append(
                    random.sample(new_list_n[2],length_zu_2)), new_new_list_n.append(random.sample(new_list_n[3],length_zu_3))
                new_new_list_n.append(random.sample(new_list_n[4],length_zu_4)), new_new_list_n.append(
                    random.sample(new_list_n[5],length_zu_5)), new_new_list_n.append(
                    random.sample(new_list_n[6],length_zu_6)), new_new_list_n.append(random.sample(new_list_n[7],length_zu_7)), new_new_list_n.append(random.sample(new_list_n[8],length_zu_8))
                new_new_list_w.append(random.sample(new_list_w[0],length_zu_0)), new_new_list_w.append(
                    random.sample(new_list_w[1],length_zu_1)), new_new_list_w.append(
                    random.sample(new_list_w[2],length_zu_2)), new_new_list_w.append(random.sample(new_list_w[3],length_zu_3))
                new_new_list_w.append(random.sample(new_list_w[4],length_zu_4)), new_new_list_w.append(
                    random.sample(new_list_w[5],length_zu_5)), new_new_list_w.append(
                    random.sample(new_list_w[6],length_zu_6)), new_new_list_w.append(random.sample(new_list_w[7],length_zu_7)), new_new_list_w.append(random.sample(new_list_w[8],length_zu_8))


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
                    chazhi = math.pow(abs(youxi - 0.19),2)  # 与最佳游隙的差值
                    fitness_1 += chazhi

                fitness_1 = fitness_1 / len(new_new_list_g)  # 优化2：游隙均值
                fitness_11 = fitness_1*100000 + hetao_F
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
                    if pro_piancha[8] < t :
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
                new_new_list_g.append(random.sample(new_list_g[0],length_zu_0)), new_new_list_g.append(
                    random.sample(new_list_g[1],length_zu_1)), new_new_list_g.append(
                    random.sample(new_list_g[2],length_zu_2)), new_new_list_g.append(random.sample(new_list_g[3],length_zu_3))
                new_new_list_g.append(random.sample(new_list_g[4],length_zu_4)), new_new_list_g.append(
                    random.sample(new_list_g[5],length_zu_5)), new_new_list_g.append(
                    random.sample(new_list_g[6],length_zu_6)), new_new_list_g.append(
                    random.sample(new_list_g[7],length_zu_7)), new_new_list_g.append(random.sample(new_list_g[8],length_zu_8)), new_new_list_g.append(random.sample(new_list_g[9],length_zu_9))
                new_new_list_n.append(random.sample(new_list_n[0],length_zu_0)), new_new_list_n.append(
                    random.sample(new_list_n[1],length_zu_1)), new_new_list_n.append(
                    random.sample(new_list_n[2],length_zu_2)), new_new_list_n.append(random.sample(new_list_n[3],length_zu_3))
                new_new_list_n.append(random.sample(new_list_n[4],length_zu_4)), new_new_list_n.append(
                    random.sample(new_list_n[5],length_zu_5)), new_new_list_n.append(
                    random.sample(new_list_n[6],length_zu_6)), new_new_list_n.append(
                    random.sample(new_list_n[7],length_zu_7)), new_new_list_n.append(random.sample(new_list_n[8],length_zu_8)), new_new_list_n.append(random.sample(new_list_n[9],length_zu_9))
                new_new_list_w.append(random.sample(new_list_w[0],length_zu_0)), new_new_list_w.append(
                    random.sample(new_list_w[1],length_zu_1)), new_new_list_w.append(
                    random.sample(new_list_w[2],length_zu_2)), new_new_list_w.append(random.sample(new_list_w[3],length_zu_3))
                new_new_list_w.append(random.sample(new_list_w[4],length_zu_4)), new_new_list_w.append(
                    random.sample(new_list_w[5],length_zu_5)), new_new_list_w.append(
                    random.sample(new_list_w[6],length_zu_6)), new_new_list_w.append(
                    random.sample(new_list_w[7],length_zu_7)), new_new_list_w.append(random.sample(new_list_w[8],length_zu_8)), new_new_list_w.append(random.sample(new_list_w[9],length_zu_9))


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
                    chazhi = math.pow(abs(youxi - 0.19),2)  # 与最佳游隙的差值
                    fitness_1 += chazhi

                fitness_1 = fitness_1 / len(new_new_list_g)  # 优化2：游隙均值
                fitness_11 = fitness_1*100000 + hetao_F
                # print("适应度", fitness_11, "目标：", yyxx, len(yyxx), hetao_T)
        time4 = datetime.datetime.now()
        # print("概率划分运行时间_3：", (time4 - time3).seconds)
        return fitness_11,pro
        # print("概率划分运行时间_3：", (time4 - time3).seconds)
        ################################################################

def copy_list(old_arr: [int]):
    new_arr = []
    for element in old_arr:
        new_arr.append(element)
    return new_arr

class Ga:
    def __init__(self):
        self.best = None
        self.individual_list = []
        self.result_list = []
        self.fitness_list = []
        self.youxi = []
        self.result_list1 = []
        self.fitness_list1 = []
        self.pro_GA = []
        self.iter_num = 500  # LNS最大迭代次数

    def cross(self):                           # 交叉
        new_gen = []
        for i in range(0, 49):
            # 父代基因
            genes1 = copy_list(self.individual_list[i].genes)
            genes2 = copy_list(self.individual_list[i + 1].genes)

            genes_1_0 = np.random.randint(0, 2, 10)
            genes_1_0 = genes_1_0.tolist()

            genes_1_1 = genes1[10:20]
            pos1_recorder_1_1 = {value: idx for idx, value in enumerate(genes_1_1)}
            genes_1_2 = genes1[20:30]
            pos1_recorder_1_2 = {value: idx for idx, value in enumerate(genes_1_2)}
            genes_1_3 = genes1[30:]
            pos1_recorder_1_3 = {value: idx for idx, value in enumerate(genes_1_3)}

            genes_2_0 = np.random.randint(0, 2, 10)
            genes_2_0 = genes_2_0.tolist()

            genes_2_1 = genes2[10:20]
            pos1_recorder_2_1 = {value: idx for idx, value in enumerate(genes_2_1)}
            genes_2_2 = genes2[20:30]
            pos1_recorder_2_2 = {value: idx for idx, value in enumerate(genes_2_2)}
            genes_2_3 = genes2[30:]
            pos1_recorder_2_3 = {value: idx for idx, value in enumerate(genes_2_3)}

            # index1 = random.randint(0, 8)           # 左闭右闭
            # index2 = random.randint(index1, 9)

            index3 = random.randint(0, 8)  # 左闭右闭
            index4 = random.randint(index3, 9)

            index5 = random.randint(0, 8)  # 左闭右闭
            index6 = random.randint(index5, 9)

            index7 = random.randint(0, 8)  # 左闭右闭
            index8 = random.randint(index7, 9)
            # 选择交叉
            kiss_num = [1,2,3]
            select = random.choice(kiss_num)
            if select == 1:
                for l in range(index3, index4):
                    value3, value4 = genes_1_1[l], genes_2_1[l]
                    pos3, pos4 = pos1_recorder_1_1[value4], pos1_recorder_2_1[value3]
                    genes_1_1[l], genes_1_1[pos3] = genes_1_1[pos3], genes_1_1[l]
                    genes_2_1[l], genes_2_1[pos4] = genes_2_1[pos4], genes_2_1[l]
                    pos1_recorder_1_1[value3], pos1_recorder_1_1[value4] = pos3, l         # 外圈片段
                    pos1_recorder_2_1[value3], pos1_recorder_2_1[value4] = l, pos4
            if select == 2:
                for o in range(index5, index6):
                    value5, value6 = genes_1_2[o], genes_2_2[o]
                    pos5, pos6 = pos1_recorder_1_2[value6], pos1_recorder_2_2[value5]
                    genes_1_2[o], genes_1_2[pos5] = genes_1_2[pos5], genes_1_2[o]
                    genes_2_2[o], genes_2_2[pos6] = genes_2_2[pos6], genes_2_2[o]
                    pos1_recorder_1_2[value5], pos1_recorder_1_2[value6] = pos5, o           # 内圈片段
                    pos1_recorder_2_2[value5], pos1_recorder_2_2[value6] = o, pos6
            if select ==3:
                for d in range(index7, index8):
                    value7, value8 = genes_1_3[d], genes_2_3[d]
                    pos7, pos8 = pos1_recorder_1_3[value8], pos1_recorder_2_3[value7]
                    genes_1_3[d], genes_1_3[pos7] = genes_1_3[pos7], genes_1_3[d]
                    genes_2_3[d], genes_2_3[pos8] = genes_2_3[pos8], genes_2_3[d]
                    pos1_recorder_1_3[value7], pos1_recorder_1_3[value8] = pos7, d           # 滚动体片段
                    pos1_recorder_2_3[value7], pos1_recorder_2_3[value8] = d, pos8
            ## 求和
            genes11111 = genes_1_0 + genes_1_1 + genes_1_2 + genes_1_3
            genes22222 = genes_2_0 + genes_2_1 + genes_2_2 + genes_2_3
            new_gen.append(Individual(genes11111))
            new_gen.append(Individual(genes22222))
        return new_gen                                                               # 198

    def mutate(self, new_gen):        # 变异
        for individual in new_gen:
            if random.random() < 0.5:
                # 翻转切片
                old_genes = copy_list(individual.genes)

                mutate_gense = np.random.randint(0, 2, 10)      # 0/1 编码的变异 使用随机产生的编码值代替
                mutate_gense = mutate_gense.tolist()
                # index1 = random.randint(0,8)
                # index2 = random.randint(index1, 9)
                genes_mutate1 = mutate_gense   # 半开区间，左闭，右开

                index3 = random.randint(10, 18)
                index4 = random.randint(index3, 19)
                genes_mutate2 = old_genes[index3:index4]
                genes_mutate2.reverse()

                # index5 = random.randint(20, 28)
                index5 = random.randint(20, 28)
                index6 = random.randint(index5, 29)
                # index6 = random.randint(index5, 29)
                genes_mutate3 = old_genes[index5:index6]
                genes_mutate3.reverse()

                index7 = random.randint(30, 38)
                index8 = random.randint(index7, 39)
                genes_mutate4 = old_genes[index7:index8]
                genes_mutate4.reverse()

                individual.genes = genes_mutate1 + old_genes[10:index3] + genes_mutate2 + old_genes[index4:20]+old_genes[20:index5] + genes_mutate3 + old_genes[index6:30]+old_genes[30:index7] + genes_mutate4 + old_genes[index8:]
        # 两代合并
        self.individual_list += new_gen                                            # 100 +198=298

    def select(self):     # 按照目标择优
        # 锦标赛
        group_num = 12  # 小组数
        group_size = 12  # 每小组人数
        group_winner = 10  # 每小组获胜人数
        winners = []  # 锦标赛结果
        for i in range(group_winner):
            group = []
            for j in range(group_size):
                # 随机组成小组
                player = random.choice(self.individual_list)
                player = Individual(player.genes)
                group.append(player)
            group = Ga.rank(group)
            # 取出获胜者
            winners += group[:5]
        self.individual_list = winners

    @staticmethod
    def rank(group):
        # 冒泡排序
        for i in range(1, len(group)):
            for j in range(0, len(group) - i):
                if group[j].fitness > group[j + 1].fitness:
                    group[j], group[j + 1] = group[j + 1], group[j]
        return group

    def destroy_operator(self, solution, num_destroy):
        # 破坏算子: 随机破坏
        a_list = [i for i in range(10)]
        k_1 = random.sample(a_list, int(num_destroy / 3))
        k_2 = random.sample(a_list, int(num_destroy / 3))
        k_3 = random.sample(a_list, int(num_destroy / 3))
        destroy_node_bank = k_1 + k_2 + k_3
        route_0 = solution.genes[:10]
        route_1 = solution.genes[10:20]
        route_2 = solution.genes[20:30]
        route_3 = solution.genes[30:40]
        for h in range(int(len(k_1))):
            route_1.remove(k_1[h])
            route_2.remove(k_2[h])
            route_3.remove(k_3[h])
        solution.genes =route_0 + route_1 + route_2 + route_3
        return solution, destroy_node_bank

    def repair_operator(self, solution, destroy_node_bank):
        # 修复算子: 随机修复
        s_route_0 = solution.genes[:10]
        s_route_1 = solution.genes[10:17]
        for w in range(3):
            in_dex_1 = random.sample([e for e in range(8 + w)], 1)
            s_route_1.insert(in_dex_1[0], destroy_node_bank[:3][w])

        s_route_2 = solution.genes[17:24]
        for n in range(3):
            in_dex_2 = random.sample([e for e in range(8 + n)], 1)
            s_route_2.insert(in_dex_2[0], destroy_node_bank[3:6][n])

        s_route_3 = solution.genes[24:]
        for g in range(3):
            in_dex_3 = random.sample([e for e in range(8 + g)], 1)
            s_route_3.insert(in_dex_3[0], destroy_node_bank[6:9][g])

        solution.genes = s_route_0 + s_route_1 + s_route_2 + s_route_3
        return solution
    def get_new_fitness(self,solution):
        Genes = solution.genes
        pro = solution.pro
        num_zu = 0  # 组数
        zero_hot = Genes[:10]  # 0/1编码确定组数，随机概率
        w_zu = Genes[10:20]  # 外圈组
        n_zu = Genes[20:30]  # 内圈组
        g_zu = Genes[30:40]  # 滚动体组
        for s in zero_hot:
            if s == 1:
                num_zu += 1
            else:
                pass
        zhen_9_1 = []

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
                    groud_3_w), list_w.append(groud_4_w), list_w.append(groud_5_w), list_w.append(
                    groud_6_w), list_w.append(groud_7_w)
                list_n.append(groud_0_n), list_n.append(groud_1_n), list_n.append(groud_2_n), list_n.append(
                    groud_3_n), list_n.append(groud_4_n), list_n.append(groud_5_n), list_n.append(
                    groud_6_n), list_n.append(groud_7_n)
                list_g.append(groud_0_g), list_g.append(groud_1_g), list_g.append(groud_2_g), list_g.append(
                    groud_3_g), list_g.append(groud_4_g), list_g.append(groud_5_g), list_g.append(
                    groud_6_g), list_g.append(groud_7_g)
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

        return fitness_11



    def next_gen(self):
        # 交叉
        new_gen = self.cross()
        # 变异
        self.mutate(new_gen)
        # 选择
        self.select()
        # LNS操作
        #==========
        num_destroy = 9
        new_individual_list = []
        mid_individual = random.sample(self.individual_list,20)
        final_ = []
        for G in self.individual_list:
            if G not in mid_individual:
                final_.append(G)
        for r_ in mid_individual:
            best_solution = r_
            for n_gen in range(self.iter_num):
                tem_solution = deepcopy(best_solution)
                # 执行破坏修复算子，得到临时解
                tem_solution, destroy_node_bank = self.destroy_operator(tem_solution, num_destroy)
                tem_solution = self.repair_operator(tem_solution, destroy_node_bank)
                # 计算临时解的目标值
                tem_solution.fitness = self.get_new_fitness(tem_solution)

                # 接受标准：如果临时解比当前解好，直接接受；且更新最优解
                if tem_solution.fitness < best_solution.fitness:
                    best_solution = deepcopy(tem_solution)
            new_individual_list.append(best_solution)
        # 更新精英群体
        self.individual_list = new_individual_list + final_

        # 获得这一代的结果
        for individual in self.individual_list:
            if individual.fitness < self.best.fitness:
                self.best = individual

    def train(self):
        # 初代种群
        self.individual_list = [Individual() for _ in range(50)]
        self.best = self.individual_list[0]
        # 迭代
        for i in range(2000):
            self.next_gen()
            result = copy_list(self.best.genes)
            print('每一代最好的基因：',result)
            print('每一代最好的适应度：',self.best.fitness)
            print('对应的分组概率：', self.best.pro)
            self.result_list.append(result)
            self.fitness_list.append(self.best.fitness)
            self.pro_GA.append(self.best.pro)
        return self.result_list, self.fitness_list,self.pro_GA
# 遗传算法运行1
time_0 = datetime.datetime.now()

ga = Ga()
result_list, fitness_list,  pro_final = ga.train()

time_1 = datetime.datetime.now()
print("LNS划分运行时间：", (time_1 - time_0).seconds)
plt.plot([i for i in range(int(len(fitness_list)))], fitness_list, 'r-')
plt.ylabel("游隙指标")
plt.xlabel("迭代次数")
plt.legend(["GA+LNS"])
plt.show()

