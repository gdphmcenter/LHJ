import math
import argparse
import numpy as np
import torch
import datetime
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from gpn import GPN

time1 = datetime.datetime.now()
# args
parser = argparse.ArgumentParser(description="GPN test")
parser.add_argument('--size', default=210, help="size of model")
parser.add_argument('--batch_size', default=64, help='')
parser.add_argument('--test_size', default=150, help="size of TSP")
parser.add_argument('--test_steps', default=1, help='')
args = vars(parser.parse_args())

B = int(args['batch_size'])
size = int(args['size'])
test_size = int(args['test_size'])
n_test = int(args['test_steps'])
load_root ='./model/gpn_tsp'+str(size)+'.pt'

print('=========================')
print('prepare to test')
print('=========================')
# print('Hyperparameters:')
# print('size', size)
# print('batch size', B)
# print('test size', test_size)
# print('test steps', n_test)
# print('load root:', load_root)
# print('=========================')

def getset():     # job_number:30 , samples:3000 每一个sample有30个工件的偏差     data_set(size):[3000,30]
    data_set = []
    zuhe = []
    for l in range(64):
        np.random.seed(0)
        for j in range(3):
            if j == 0:
                job_w_piancha = np.random.normal(loc=22.5, scale=7.5, size=70)
                zuhe.extend(job_w_piancha)
            if j == 1:
                job_n_piancha = np.random.normal(loc=-24, scale=2, size=70)
                zuhe.extend(job_n_piancha)
            if j == 2:
                job_g_piancha = np.random.normal(loc=10.75, scale=2.75, size=70)
                zuhe.extend(job_g_piancha)
    zuhe = np.array(zuhe)
    zuhe = zuhe.reshape(64,210,1)          # batch、size、value
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
    
    X = getset()
    
    X = torch.Tensor(X)
    
    mask = torch.zeros(B,test_size)
    
    R = 0
    Idx = []
    reward = 0
    
    Y = X.view(B,test_size,1)           # to the same batch size
    x = Y[:,0,:]
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
            batch_reward_j.append(batch_list2[64 * Q + j])
        w_j = []
        n_j = []
        g_j = []
        for r in batch_reward_j:
            if r < torch.tensor(70):
                w_j.append(r)
            if torch.tensor(69) < r < torch.tensor(140):
                n_j.append(r)
            if r > torch.tensor(139):
                g_j.append(r)
        batch_j_input = Y[j, :, :]
        batch_j_input = batch_j_input.view(210)
        youxi = 0
        T = 0
        RRR = []
        for t in range(70):
            youxi_true = (batch_j_input[int(w_j[t])] - batch_j_input[int(n_j[t])] - 2 * batch_j_input[int(g_j[t])])
            RRR.append(youxi_true)
            youxi += torch.tensor(math.pow(abs(torch.tensor(25) - youxi_true), 2))
            # youxi += abs(torch.tensor(25) - youxi_true)
            if torch.tensor(10) < youxi_true < torch.tensor(40):
                T += 1
        # print(RRR)
        hetao1.append(T)
        youxi = youxi / torch.tensor(70)
        youxi_num_batch2.append(youxi)
    R = torch.tensor(youxi_num_batch2)

    tour_len += R.mean().item()

    print('test:{}, 平均游隙:{}'.format(m, tour_len),"平均合套率", (sum(hetao1) / len(hetao1)) / 70)
    
    total_len += tour_len
    tor_len_hetaolv += (sum(hetao1) / len(hetao1)) / 70
time2 = datetime.datetime.now()
print('total 游隙均值:', total_len/n_test, "total 平均合套率：",tor_len_hetaolv/n_test)
print('running time_batch_64:', (time2-time1).seconds,"秒")
print('running time_batch_1:', ((time2-time1).seconds)/64, "秒")