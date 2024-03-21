import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import math
from numpy import *


# 变异概率
np.random.seed(0)
w = np.random.normal(loc=22.5,scale=7.5,size=70)
n = np.random.normal(loc=-24,scale=2,size=70)
g = np.random.normal(loc=10.75,scale=2.75,size=70)
geti_Norm = np.concatenate((w,n,g),axis=0)
class Individual:
    def __init__(self, genes=None):
        # 随机生成序
        if genes is None:
            genes = [i for i in geti_Norm]
            gense1,gense2,gense3 = genes[:70],genes[70:140],genes[140:210]
            random.shuffle(gense1)
            random.shuffle(gense2)
            random.shuffle(gense3)
            genes = gense1+gense2+gense3
        self.genes = genes
        self.fitness = self.evaluate_fitness()
    def evaluate_fitness(self):
        # 计算个体适应度
        fitness = []
        for i in range(70):
            w_idx = self.genes[i]
            n_idx = self.genes[i + 70]
            g_idx = self.genes[i + 140]
            fitness1 = w_idx - n_idx - 2 * g_idx
            fitness1 = abs(fitness1-25)
            fitness1 = math.pow(fitness1,2)
            fitness.append(fitness1)
        avg_fitness = sum(fitness) / 70
        return avg_fitness

def copy_list(old_arr: [int]):
    new_arr = []
    for element in old_arr:
        new_arr.append(element)
    return new_arr

class Ga:
    def __init__(self):
        self.best = None
        self.pro_mutate = 0.4
        self.individual_list = []
        self.result_list = []
        self.fitness_list = []
        self.result_list1 = []
        self.fitness_list1 = []
    def cross(self):                           # 交叉
        new_gen = []
        for i in range(0, 99):
            # 父代基因
            genes1 = copy_list(self.individual_list[i].genes)
            genes2 = copy_list(self.individual_list[i + 1].genes)
            index1 = random.randint(0, 68)           # 左闭右闭
            index2 = random.randint(index1, 69)
            index3 = random.randint(70, 138)  # 左闭右闭
            index4 = random.randint(index3, 139)
            index5 = random.randint(140, 208)  # 左闭右闭
            index6 = random.randint(index5, 209)
            pos1_recorder = {value: idx for idx, value in enumerate(genes1)}
            pos2_recorder = {value: idx for idx, value in enumerate(genes2)}
            # 交叉
            kiss_num =[1,2,3]
            a = random.choice(kiss_num)
            if a == 1:
                for j in range(index1, index2):
                    value1, value2 = genes1[j], genes2[j]
                    pos1, pos2 = pos1_recorder[value2], pos2_recorder[value1]
                    genes1[j], genes1[pos1] = genes1[pos1], genes1[j]
                    genes2[j], genes2[pos2] = genes2[pos2], genes2[j]
                    pos1_recorder[value1], pos1_recorder[value2] = pos1, j
                    pos2_recorder[value1], pos2_recorder[value2] = j, pos2
            if a ==2:
                for l in range(index3, index4):
                    value3, value4 = genes1[l], genes2[l]
                    pos3, pos4 = pos1_recorder[value4], pos2_recorder[value3]
                    genes1[l], genes1[pos3] = genes1[pos3], genes1[l]
                    genes2[l], genes2[pos4] = genes2[pos4], genes2[l]
                    pos1_recorder[value3], pos1_recorder[value4] = pos3, l
                    pos2_recorder[value3], pos2_recorder[value4] = l, pos4
            if a ==3:
                for o in range(index5, index6):
                    value5, value6 = genes1[o], genes2[o]
                    pos5, pos6 = pos1_recorder[value6], pos2_recorder[value5]
                    genes1[o], genes1[pos5] = genes1[pos5], genes1[o]
                    genes2[o], genes2[pos6] = genes2[pos6], genes2[o]
                    pos1_recorder[value5], pos1_recorder[value6] = pos5, o
                    pos2_recorder[value5], pos2_recorder[value6] = o, pos6
            new_gen.append(Individual(genes1))
            new_gen.append(Individual(genes2))
        return new_gen                                                               # 198

    def mutate(self, new_gen):        # 变异
        for individual in new_gen:
            if random.random() < self.pro_mutate:
                # 翻转切片
                old_genes = copy_list(individual.genes)
                index1 = random.randint(0,68)
                index2 = random.randint(index1, 69)
                genes_mutate1 = old_genes[index1:index2]    # 半开区间，左闭，右开
                genes_mutate1.reverse()

                index3 = random.randint(70, 138)
                index4 = random.randint(index3, 139)
                genes_mutate2 = old_genes[index3:index4]
                genes_mutate2.reverse()

                index5 = random.randint(140, 208)
                index6 = random.randint(index5, 209)
                genes_mutate3 = old_genes[index5:index6]
                genes_mutate3.reverse()

                individual.genes = old_genes[0:index1] + genes_mutate1 + old_genes[index2:70]+old_genes[70:index3] + genes_mutate2 + old_genes[index4:140]+old_genes[140:index5] + genes_mutate3 + old_genes[index6:]
        # 两代合并
        self.individual_list += new_gen                                            # 100 +198=298

    def select(self):     # 按照目标择优
        # 锦标赛
        group_num = 15  # 小组数
        group_size = 15  # 每小组人数
        group_winner = 10  # 每小组获胜人数
        winners = []  # 锦标赛结果
        for i in range(group_winner):
            group = []
            for j in range(group_size):
                # 随机组成小组
                player = random.choice(self.individual_list)
                player = Individual(player.genes)
                                                                                     # 289 选 225  进 100
                group.append(player)
            group = Ga.rank(group)
            # 取出获胜者
            winners += group[:group_winner]
        self.individual_list = winners

    @staticmethod
    def rank(group):
        # 冒泡排序
        for i in range(1, len(group)):
            for j in range(0, len(group) - i):
                if group[j].fitness > group[j + 1].fitness:
                    group[j], group[j + 1] = group[j + 1], group[j]
        return group

    def next_gen(self):
        # 交叉
        new_gen = self.cross()
        # 变异
        self.mutate(new_gen)
        # 选择
        self.select()
        # 获得这一代的结果
        for individual in self.individual_list:
            if individual.fitness < self.best.fitness:
                self.best = individual

    def train(self):
        # 初代种群
        self.individual_list = [Individual() for _ in range(100)]
        self.best = self.individual_list[0]
        # 迭代
        for i in range(200000):
            T = 0
            self.next_gen()
            result = copy_list(self.best.genes)
            self.result_list.append(result)
            self.fitness_list.append(self.best.fitness)
            # if len(self.fitness_list) > 1:
            #     self.fitness_list.reverse()
            #     for Y in self.fitness_list:
            #         if Y == self.fitness_list[0]:
            #             T += 1
            #         else:
            #             break
            #     if 500 <= T < 1500:
            #         self.pro_mutate = 0.2
            #         self.fitness_list.reverse()
            #     elif 1500 <= T < 2500:
            #         self.pro_mutate = 0.3
            #         self.fitness_list.reverse()
            #     elif 2500 <= T < 3500:
            #         self.pro_mutate = 0.4
            #         self.fitness_list.reverse()
            #     elif 3500 <= T < 9999999:
            #         self.pro_mutate = 0.5
            #         self.fitness_list.reverse()
            #         # for t in range(1,50):
            #         #     if T//3500 == t :
            #         # # self.individual_list = []
            #         # # self.individual_list = [Individual() for _ in range(100)]
            #     else:
            #         self.pro_mutate = 0.05
            #         self.fitness_list.reverse()
        return self.result_list, self.fitness_list


# 遗传算法运行1
#############################
ga = Ga()
result_list, fitness_list = ga.train()
steps = [i for i in range(200000)]
plt.plot(steps, fitness_list)
plt.show()
result = result_list[-1]
value = []
n_element = []
n_value = []
compare = 0
compare_sqrt = 0
T = 0
for b in range(70):
    value1 = result[b]-result[b+70] - 2*result[b+140]
    if 10 < value1 < 40:
        T+=1
    value.append(value1)
print('每一组对应的组合的真实游隙值',value)
for a in value:
    compare_sqrt += math.pow(abs(a-25),2)                                                  # 奖励值修改为差值的平方
    compare += abs(a-25)
print("平均游隙差值：",compare/70, "合套率：",T/70)
print("平均游隙差值(平方)：",compare_sqrt/70, "合套率：",T/70)
