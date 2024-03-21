import random
import math
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import  time

# 参数设定
group = 70   # 组数
seed = 3     # 随机种子
step = 100000 # 迭代次数


class PSO(object):
    def __init__(self, num_city, data):
        self.youxi = []  # 记录游隙
        self.p = []  # 合套方案

        self.iter_max = step  # 迭代数目
        self.num = 300  # 粒子数目
        self.num_city = num_city  # 城市数
        self.location = data# 城市的位置坐标            # (150,1)
        # 计算距离矩阵
        # 初始化所有粒子
        # self.particals = self.random_init(self.num, num_city)
        self.particals = self.greedy_init(num_total=self.num,num_city=num_city)
        self.lenths = self.compute_paths(self.particals,self.location)
        # 得到初始化群体的最优解
        init_l = min(self.lenths)
        init_index = self.lenths.index(init_l)
        init_path = self.particals[init_index]
        # 画出初始的路径图
        init_show = self.location[init_path]
        # 记录每个个体的当前最优解
        self.local_best = self.particals
        self.local_best_len = self.lenths
        # 记录当前的全局最优解,长度是iteration
        self.global_best = init_path
        self.global_best_len = init_l
        # 输出解
        self.best_l = self.global_best_len
        self.best_path = self.global_best
        # 存储每次迭代的结果，画出收敛图
        self.iter_x = [0]
        self.iter_y = [init_l]


    def greedy_init(self, num_total, num_city):
        result = []
        for i in range(num_total):
            q1 = [i for i in range(num_city)]
            random.shuffle(q1)
            result.append(q1)
        return result

    # def greedy_init(self, dis_mat, num_total, num_city):
    #     start_index = 0
    #     result = []
    #     for i in range(num_total):
    #         rest = [x for x in range(0, num_city)]
    #         # 所有起始点都已经生成了
    #         if start_index >= num_city:
    #             start_index = np.random.randint(0, num_city)
    #             result.append(result[start_index].copy())
    #             continue
    #         current = start_index
    #         rest.remove(current)
    #         # 找到一条最近邻路径
    #         result_one = [current]
    #         while len(rest) != 0:
    #             tmp_min = math.inf
    #             tmp_choose = -1
    #             for x in rest:
    #                 if dis_mat[current][x] < tmp_min:
    #                     tmp_min = dis_mat[current][x]
    #                     tmp_choose = x
    #
    #             current = tmp_choose
    #             result_one.append(tmp_choose)
    #             rest.remove(tmp_choose)
    #         result.append(result_one)
    #         start_index += 1
    #     return result

    # 随机初始化
    def random_init(self, num_total, num_city):
        tmp = [x for x in range(num_city)]
        result = []
        for i in range(num_total):
            random.shuffle(tmp)
            result.append(tmp.copy())
        return result



    # 计算一条路径的长度
    def compute_pathlen(self, path,size):
        # w/n/g排序
        size = list(size)
        w = []
        n = []
        g = []
        for u in path:
            if u < group:
                w.append(u)
            if group-1 < u < 2*group:
                n.append(u)
            if u > 2*group-1:
                g.append(u)
        youxi_cha_sqrt = 0
        for i in range(int(len(path)/3)):
            y_x_true = float(size[w[i]])-float(size[n[i]])-2*float(size[g[i]])
            youxi_cha_sqrt += math.pow(abs(y_x_true-25),2)
        result = youxi_cha_sqrt
        # self.youxi.append(result/50)
        # self.p.append(str(path))
        return result/group

    # 计算一个群体的长度
    def compute_paths(self, paths, size):
        result = []
        for one in paths:
            length = self.compute_pathlen(one,size)
            result.append(length)
        return result

    # 评估当前的群体
    def eval_particals(self):
        min_lenth = min(self.lenths)
        min_index = self.lenths.index(min_lenth)
        cur_path = self.particals[min_index]
        # w = []
        # n = []
        # g = []
        # t = 0
        # for u in cur_path:
        #     if u < 50:
        #         w.append(u)
        #     if 49 < u < 100:
        #         n.append(u)
        #     if u > 99 :
        #         g.append(u)
        # youxi_cha_sqrt = 0
        # for i in range(int(len(self.global_best) / 3)):
        #     y_x_true = float(list(self.location)[w[i]]) - float(list(self.location)[n[i]]) - 2 * float(list(self.location)[g[i]])
        #     youxi_cha_sqrt += math.pow(abs(y_x_true - 25), 2)
        #     if 10 < y_x_true < 40:
        #         t += 1
        # print(youxi_cha_sqrt / 50, t / 50)
        # print(min_lenth)
        # 更新当前的全局最优
        if min_lenth < self.global_best_len:
            self.global_best_len = min_lenth
            self.global_best = cur_path
            # w = []
            # n = []
            # g = []
            # t = 0
            # for u in self.global_best:
            #     if u < 50:
            #         w.append(u)
            #     if 49 < u < 100:
            #         n.append(u)
            #     if u > 99 :
            #         g.append(u)
            # youxi_cha_sqrt = 0
            # for i in range(int(len(self.global_best) / 3)):
            #     y_x_true = float(list(self.location)[w[i]]) - float(list(self.location)[n[i]]) - 2 * float(list(self.location)[g[i]])
            #     youxi_cha_sqrt += math.pow(abs(y_x_true - 25), 2)
            #     if 10 < y_x_true < 40:
            #         t += 1
            # print(youxi_cha_sqrt / 50, t / 50)
            # print(self.global_best_len)

        # 更新当前的个体最优
        for i, l in enumerate(self.lenths):
            if l < self.local_best_len[i]:
                self.local_best_len[i] = l
                self.local_best[i] = self.particals[i]

    # 粒子交叉
    def cross(self, cur, best):
        one = cur.copy()
        l = [t for t in range(self.num_city)]
        t = np.random.choice(l,2)
        x = min(t)
        y = max(t)
        cross_part = best[x:y]
        tmp = []
        for t in one:
            if t in cross_part:
                continue
            tmp.append(t)
        # 两种交叉方法
        one = tmp + cross_part
        l1 = self.compute_pathlen(one, self.location)
        one2 = cross_part + tmp
        l2 = self.compute_pathlen(one2, self.location)
        if l1<l2:
            return one, l1
        else:
            return one, l2


    # 粒子变异
    def mutate(self, one):
        one = one.copy()
        l = [t for t in range(self.num_city)]
        t = np.random.choice(l, 2)
        x, y = min(t), max(t)
        one[x], one[y] = one[y], one[x]
        l2 = self.compute_pathlen(one,self.location)
        return one, l2

    # 迭代操作
    def pso(self,):
        for cnt in range(1, self.iter_max):

            # 更新粒子群
            for i, one in enumerate(self.particals):
                tmp_l = self.lenths[i]
                # 与当前个体局部最优解进行交叉
                new_one, new_l = self.cross(one, self.local_best[i])
                if new_l < self.best_l:
                    self.best_l = tmp_l
                    self.best_path = one

                if new_l < tmp_l or np.random.rand() < 0.1:
                    one = new_one
                    tmp_l = new_l

                # 与当前全局最优解进行交叉
                new_one, new_l = self.cross(one, self.global_best)

                if new_l < self.best_l:
                    self.best_l = tmp_l
                    self.best_path = one

                if new_l < tmp_l or np.random.rand() < 0.1:
                    one = new_one
                    tmp_l = new_l
                # 变异
                one, tmp_l = self.mutate(one)

                if new_l < self.best_l:
                    self.best_l = tmp_l
                    self.best_path = one

                if new_l < tmp_l or np.random.rand() < 0.1:
                    one = new_one
                    tmp_l = new_l

                # 更新该粒子
                self.particals[i] = one
                self.lenths[i] = tmp_l
            # 评估粒子群，更新个体局部最优和个体当前全局最优
            self.eval_particals()

            # w = []
            # n = []
            # g = []
            # t = 0
            # for u in self.best_path:
            #     if u < 50:
            #         w.append(u)
            #     if 49 < u < 100:
            #         n.append(u)
            #     if u > 99 :
            #         g.append(u)
            # youxi_cha_sqrt = 0
            # for i in range(int(len(self.best_path) / 3)):
            #     y_x_true = float(list(self.location)[w[i]]) - float(list(self.location)[n[i]]) - 2 * float(list(self.location)[g[i]])
            #     youxi_cha_sqrt += math.pow(abs(y_x_true - 25), 2)
            #     if 10 < y_x_true < 40:
            #         t += 1
            # print(youxi_cha_sqrt / 50, t / 50)
            # print(cnt ,self.best_l)


            # 更新输出解
            size = self.location
            size = list(size)
            if self.global_best_len < self.best_l:
                self.best_l = self.global_best_len
                self.best_path = self.global_best
            # w = []
            # n = []
            # g = []
            # t = 0
            # for u in self.best_path:
            #     if u < 50:
            #         w.append(u)
            #     if 49 < u < 100:
            #         n.append(u)
            #     if u > 99:
            #         g.append(u)
            # youxi_cha_sqrt = 0
            # for i in range(int(len(self.best_path) / 3)):
            #     y_x_true = float(size[w[i]]) - float(size[n[i]]) - 2 * float(size[g[i]])
            #     youxi_cha_sqrt += math.pow(abs(y_x_true - 25), 2)
            #     if 10 < y_x_true < 40:
            #         t += 1
            # print(youxi_cha_sqrt / 50, t / 50)
            # print(self.best_l,self.best_path)
            self.iter_x.append(cnt)
            self.iter_y.append(self.best_l)
        # 返回游隙差值最小值的下标
        # min_index = self.youxi.index(min(self.youxi))
        # min_hetao = self.p[min_index]

        return self.best_l, self.best_path

    def run(self):
        best_length, best_path = self.pso()
        # 画出最终路径
        return best_path, best_length,


# 读取数据
# def read_tsp(path):
#     lines = open(path, 'r').readlines()
#     assert 'NODE_COORD_SECTION\n' in lines
#     index = lines.index('NODE_COORD_SECTION\n')
#     data = lines[index + 1:-1]
#     tmp = []
#     for line in data:
#         line = line.strip().split(' ')
#         if line[0] == 'EOF':
#             continue
#         tmpline = []
#         for x in line:
#             if x == '':
#                 continue
#             else:
#                 tmpline.append(x)
#         if tmpline == []:
#             continue
#         tmp.append(tmpline)
#     data = tmp
#     return data


np.random.seed(seed)
w = np.random.normal(loc=22.5,scale=7.5,size=group)
n = np.random.normal(loc=-24,scale=2,size=group)
g = np.random.normal(loc=10.75,scale=2.75,size=group)
geti_Norm = np.concatenate((w,n,g),axis=0)
data = geti_Norm.reshape(3*group)




startT = time.perf_counter()
model = PSO(num_city=data.shape[0], data=data.copy())
Best_path, Best = model.run()
endT = time.perf_counter()
runTime = endT - startT
print("PSO"+str(len(list(data))/3)+"规模的运行时间为"+str(round(runTime,2)))
# print(Best)
# print(hetao)
# w = []
# n = []
# g = []
# t = 0
# for u in hetao:
#     if u < 50:
#         w.append(u)
#     if 49 < u < 100:
#         n.append(u)
#     if u > 99:
#         g.append(u)
# youxi_cha_sqrt = 0
# for i in range(int(50)):
#     y_x_true = float(list(data)[w[i]]) - float(list(data)[n[i]]) - 2 * float(list(data)[g[i]])
#     youxi_cha_sqrt += math.pow(abs(y_x_true - 25), 2)
#     if 10 < y_x_true < 40:
#         t += 1
# print(youxi_cha_sqrt / 50, t / 50)
# size = list(data)
# w = []
# n = []
# g = []
# t = 0
# for u in Best_path:
#     if u < 50:
#         w.append(u)
#     if 49 < u < 100:
#         n.append(u)
#     if u > 99:
#         g.append(u)
# youxi_cha_sqrt = 0
# for i in range(int(len(Best_path)/3)):
#     y_x_true = float(size[w[i]])-float(size[n[i]])-2*float(size[g[i]])
#     youxi_cha_sqrt += math.pow(abs(y_x_true-25),2)
#     if 10 < y_x_true < 40:
#         t += 1
#
# result = youxi_cha_sqrt
# print("平均游隙差值平方",result/50,"合套率",t/50)
print("迭代完平均游隙差值平方", Best)
fig, axs = plt.subplots(2, 1, sharex=False, sharey=False)
iterations = model.iter_x
best_record = model.iter_y
axs[1].plot(iterations, best_record)
axs[1].set_title('收敛曲线')
plt.show()


