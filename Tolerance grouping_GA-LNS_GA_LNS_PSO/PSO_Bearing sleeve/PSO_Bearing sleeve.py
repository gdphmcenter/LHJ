import random
import math
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei']
import  time
import datetime
import scipy.stats as st
# 参数设定

step = 2000 # 迭代次数



class PSO(object):
    def __init__(self,data):
        self.youxi = []  # 记录游隙
        self.p = []  # 合套方案

        self.iter_max = step  # 迭代数目
        self.num = 300  # 粒子数目

        self.location = data # 城市的位置坐标            # (150,1)

        # 计算距离矩阵
        # 初始化所有粒子
        self.particals = self.random_init(self.num)
        self.lenths, self.pro = self.compute_paths_all(self.particals,self.location)
        # 得到初始化群体的最优解
        init_l = min(self.lenths)
        init_index = self.lenths.index(init_l)
        init_path = self.particals[init_index]
        init_pro = self.pro[init_index]

        # 记录每个个体的当前最优解
        self.local_best = self.particals
        self.local_best_len = self.lenths
        self.local_best_pro = self.pro

        # 记录当前的全局最优解,长度是iteration
        self.global_best = init_path
        self.global_best_len = init_l
        self.global_best_pro = init_pro
        # 输出解
        self.best_l = self.global_best_len
        self.best_path = self.global_best
        self.best_pro = self.global_best_pro
        # 存储每次迭代的结果，画出收敛图
        self.iter_x = [init_path]
        self.iter_y = [init_l]
        self.iter_z = [init_pro]



    # 随机初始化
    def random_init(self,num):
        genes_list = []
        for r in range(num):
            z_h = np.random.randint(0, 2, 10).tolist()
            a_list = [i for i in range(10)]
            b_list = [i for i in range(10)]
            c_list = [i for i in range(10)]
            random.shuffle(a_list)
            random.shuffle(b_list)
            random.shuffle(c_list)
            z_h.extend(a_list)
            z_h.extend(b_list)
            z_h.extend(c_list)

            genes_list.append(z_h)
        return genes_list

    def compute_paths_one(self, genes, data):
        w = data[:1000]
        n = data[1000:2000]
        g = data[2000:3000]

        # 计算个体适应度
        num_zu = 0  # 组数
        zero_hot = genes[:10]  # 0/1编码确定组数，随机概率
        w_zu = genes[10:20]  # 外圈组
        n_zu = genes[20:30]  # 内圈组
        g_zu = genes[30:40]  # 滚动体组
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


        if len(pro_w) > 1:
            for P in range(1, len(pro_w)):
                pro_piancha.append(st.norm.ppf(sum(pro_w[:P]), loc=308.92, scale=0.00500))
                pro_piancha_n.append(st.norm.ppf(sum(pro_n[:P]), loc=244.745, scale=0.01333))
                pro_piancha_g.append(st.norm.ppf(sum(pro_g[:P]), loc=31.9925, scale=0.00417))

        length = len(pro_w)

        for z in range(11):  # 0-10 闭区间                  # 统一返回一个变量，再在最后用return
            if z == 0 and z == length:  # 0组
                fitness_11 = 999999999  # 可以给一个很大的值，来淘汰掉
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

        return fitness_11, pro


    # 计算一个群体的长度
    def compute_paths_all(self, genes, data):
        w = data[:1000]
        n = data[1000:2000]
        g = data[2000:3000]
        youxi_list = []
        pro_list = []
        for t_t in range(300):
            # 计算个体适应度
            num_zu = 0  # 组数
            zero_hot = genes[t_t][:10]  # 0/1编码确定组数，随机概率
            w_zu = genes[t_t][10:20]  # 外圈组
            n_zu = genes[t_t][20:30]  # 内圈组
            g_zu = genes[t_t][30:40]  # 滚动体组
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


            if len(pro_w) > 1:
                for P in range(1, len(pro_w)):
                    pro_piancha.append(st.norm.ppf(sum(pro_w[:P]), loc=308.92, scale=0.00500))
                    pro_piancha_n.append(st.norm.ppf(sum(pro_n[:P]), loc=244.745, scale=0.01333))
                    pro_piancha_g.append(st.norm.ppf(sum(pro_g[:P]), loc=31.9925, scale=0.00417))

            length = len(pro_w)

            for z in range(11):  # 0-10 闭区间                  # 统一返回一个变量，再在最后用return
                if z == 0 and z == length:  # 0组
                    fitness_11 = 999999999  # 可以给一个很大的值，来淘汰掉
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
            youxi_list.append(fitness_11)
            pro_list.append(pro)

        return youxi_list, pro_list





        return result

    # 评估当前的群体
    def eval_particals(self):
        min_lenth = min(self.lenths)
        min_index = self.lenths.index(min_lenth)
        cur_path = self.particals[min_index]
        cur_pro = self.pro[min_index]

        if min_lenth < self.global_best_len:
            self.global_best_len = min_lenth
            self.global_best = cur_path
            self.global_best_pro = cur_pro


        # 更新当前的个体最优
        for i, l in enumerate(self.lenths):
            if l < self.local_best_len[i]:
                self.local_best_len[i] = l
                self.local_best[i] = self.particals[i]
                self.local_best_pro[i] = self.pro[i]

    # 粒子交叉
    def cross(self, cur, best):
        one_ = cur.copy()
        # 外交叉
        l_1 = [t for t in range(10)]
        t_1 = random.sample(l_1,2)
        x_1 = min(t_1)
        y_1 = max(t_1)
        cross_part_1 = best[10:20][x_1:y_1]
        tmp_1 = []
        for t in one_[10:20]:
            if t in cross_part_1:
                continue
            tmp_1.append(t)
        o_1 = tmp_1 + cross_part_1
        # 内交叉
        l_2 = [t for t in range(10)]
        t_2 = random.sample(l_2, 2)
        x_2 = min(t_2)
        y_2 = max(t_2)
        cross_part_2 = best[20:30][x_2:y_2]
        tmp_2 = []
        for t in one_[20:30]:
            if t in cross_part_2:
                continue
            tmp_2.append(t)
        o_2 = tmp_2 + cross_part_2
        # 滚交叉
        l_3 = [t for t in range(10)]
        t_3 = random.sample(l_3, 2)
        x_3 = min(t_3)
        y_3 = max(t_3)
        cross_part_3 = best[30:40][x_3:y_3]
        tmp_3 = []
        for t in one_[30:40]:
            if t in cross_part_3:
                continue
            tmp_3.append(t)
        o_3 = tmp_3 + cross_part_3

        one = one_[:10]+o_1+o_2+o_3

        l1 = self.compute_paths_one(one, self.location)

        return one, l1[0], l1[1]


    # 粒子变异
    def mutate(self, one):
        one = one.copy()

        #外变异
        one_1 = one[10:20]
        l_1 = [t for t in range(10)]
        t_1 = random.sample(l_1, 2)
        x_1, y_1 = min(t_1), max(t_1)
        one_1[x_1], one_1[y_1] = one_1[y_1], one_1[x_1]
        # 内变异
        one_2 = one[20:30]
        l_2 = [t for t in range(10)]
        t_2 = random.sample(l_2, 2)
        x_2, y_2 = min(t_2), max(t_2)
        one_2[x_2], one_2[y_2] = one_2[y_2], one_2[x_2]
        # 滚变异
        one_3 = one[30:40]
        l_3 = [t for t in range(10)]
        t_3 = random.sample(l_3, 2)
        x_3, y_3 = min(t_3), max(t_3)
        one_3[x_3], one_3[y_3] = one_3[y_3], one_3[x_3]

        one = one[:10]+ one_1 + one_2 +one_3
        l2 = self.compute_paths_one(one,self.location)
        return one, l2[0],l2[1]

    # 迭代操作
    def pso(self,):
        for cnt in range(1, self.iter_max):

            # 更新粒子群
            for i, one in enumerate(self.particals):
                tmp_l = self.lenths[i]
                tmp_pro = self.pro[i]
                # 与当前个体局部最优解进行交叉
                new_one, new_l, new_pro = self.cross(one, self.local_best[i])
                if new_l < self.best_l:
                    self.best_l = tmp_l
                    self.best_path = one
                    self.best_pro = tmp_pro

                if new_l < tmp_l or np.random.rand() < 0.1:
                    one = new_one
                    tmp_l = new_l
                    tmp_pro = new_pro

                # 与当前全局最优解进行交叉
                new_one, new_l, new_pro = self.cross(one, self.global_best)

                if new_l < self.best_l:
                    self.best_l = tmp_l
                    self.best_path = one
                    self.best_pro = tmp_pro

                if new_l < tmp_l or np.random.rand() < 0.1:
                    one = new_one
                    tmp_l = new_l
                    tmp_pro = new_pro
                # 变异
                one, tmp_l,tmp_pro = self.mutate(one)

                if new_l < self.best_l:
                    self.best_l = tmp_l
                    self.best_path = one
                    self.best_pro = tmp_pro

                if new_l < tmp_l or np.random.rand() < 0.1:
                    one = new_one
                    tmp_l = new_l
                    tmp_pro = new_pro

                # 更新该粒子
                self.particals[i] = one
                self.lenths[i] = tmp_l
                self.pro[i] = tmp_pro
            # 评估粒子群，更新个体局部最优和个体当前全局最优
            self.eval_particals()


            # 更新输出解

            if self.global_best_len < self.best_l:
                self.best_l = self.global_best_len
                self.best_path = self.global_best
                self.best_pro = self.global_best_pro

            print("当前代编码：",self.best_path)
            print("当前代游隙指标：",self.best_l)
            print("当前代对应的分组概率：",self.best_pro)

            self.iter_x.append(self.best_path)
            self.iter_y.append(self.best_l)
            self.iter_z.append(self.best_pro)

        return self.iter_x, self.iter_y,self.iter_z

    def run(self):
        best_path , best_length, best_pro = self.pso()
        # 画出最终路径
        return best_path , best_length, best_pro





np.random.seed(0)
w = np.random.normal(loc=308.92, scale=0.00500, size=1000)
n = np.random.normal(loc=244.745, scale=0.01333, size=1000)
g = np.random.normal(loc=31.9925, scale=0.00417, size=1000)
geti_Norm = np.concatenate((w,n,g),axis=0)
data = geti_Norm.reshape(3000)


time_0 = datetime.datetime.now()

model = PSO(data)
Genes, youxi, ppro = model.run()
print("最终代编码：", Genes[-1])
print("最终代游隙指标：", youxi[-1])
print("最终代对应的分组概率：", ppro[-1])

time_1 = datetime.datetime.now()
print("LNS划分运行时间：", (time_1 - time_0).seconds)
youxi.sort(reverse=True)
plt.plot([i for i in range(int(len(youxi)))], youxi, 'r-')
plt.ylabel("游隙指标")
plt.xlabel("迭代次数")
plt.legend(["PSO"])
plt.show()



