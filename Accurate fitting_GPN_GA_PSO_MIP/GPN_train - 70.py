import math
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.optim import lr_scheduler
import matplotlib.pyplot as plt
from tqdm import tqdm
from gpn import GPN

def getset():     # job_number:30 , samples:3000 每一个sample有30个工件的偏差     data_set(size):[3000,30]
    data_set = []
    zuhe = []
    for l in range(64):
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


if __name__ == "__main__":

    # args
    parser = argparse.ArgumentParser(description="GPN with RL")
    parser.add_argument('--size', default=210, help="size of TSP")
    parser.add_argument('--epoch', default=20, help="number of epochs")
    parser.add_argument('--batch_size', default=64, help='')
    parser.add_argument('--train_size', default=2000, help='')
    parser.add_argument('--val_size', default=64, help='')
    parser.add_argument('--lr', type=float, default=0.0005, help="learning rate")
    args = vars(parser.parse_args())

    size = int(args['size'])
    learn_rate = args['lr']    # learning rate
    B = int(args['batch_size'])    # batch_size
    B_val = int(args['val_size'])    # validation size
    steps = int(args['train_size'])    # training steps
    n_epoch = int(args['epoch'])    # epochs
    save_root ='./model/gpn_tsp'+str(size)+'.pt'
    
    print('=========================')
    print('prepare to train')
    print('=========================')
    print('Hyperparameters:')
    print('size', size)
    print('learning rate', learn_rate)
    print('batch size', B)
    print('validation size', B_val)
    print('steps', steps)
    print('epoch', n_epoch)
    print('save root:', save_root)
    print('=========================')
    
    
    model = GPN(n_feature=1, n_hidden=128)
    # load model
    # model = torch.load(save_root).cuda()
    optimizer = optim.Adam(model.parameters(), lr=learn_rate)

    lr_decay_step = 2500
    lr_decay_rate = 0.96
    opt_scheduler = lr_scheduler.MultiStepLR(optimizer, range(lr_decay_step, lr_decay_step*1000,
                                         lr_decay_step), gamma=lr_decay_rate)
    
    # validation data
    # X_val = np.random.rand(B_val, size, 2)


    C = 0     # baseline
    R = 0     # reward

    # R_mean = []
    # R_std = []
    for epoch in range(n_epoch):
        for i in tqdm(range(steps)):
            optimizer.zero_grad()

            X_roll = getset()
            X_roll = torch.Tensor(X_roll)
            mask = torch.zeros(B,210)
        
            R = 0
            logprobs = 0
            reward = 0
            
            Y = X_roll.view(B,210,1)
            x = Y[:,0,:]

            h = None
            c = None
            batch_list = []
            for k in range(size):
                
                output, h, c, _ = model(x=x, X_all=X_roll, h=h, c=c, mask=mask)     # x、mask1不同，h、c相同
                
                sampler = torch.distributions.Categorical(output)
                idx = sampler.sample()         # now the idx has B elements
                Y1 = Y[[i for i in range(B)], idx.data].clone()

                # TSP问题奖励值计算

                # if k == 0:
                #     Y_ini = Y1.clone()
                # if k > 0:
                #     reward = torch.norm(Y1-Y0, dim=1)
                #
                # Y0 = Y1.clone()
                # x = Y[[i for i in range(B)], idx.data].clone()
                #
                # R += reward


                # 合套计算游隙、batch
                batch_list.extend(idx)

                TINY = 1e-15
                logprobs += torch.log(output[[i for i in range(B)], idx.data]+TINY)
                mask[[i for i in range(B)], idx.data] += -np.inf

            youxi_num_batch = []
            hetao = []
            for j in range(B):
                batch_reward_j = []
                for Q in range(size):  # size:150
                    batch_reward_j.append(batch_list[64*Q + j])
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
                for t in range(70):
                    youxi_true = batch_j_input[int(w_j[t])] - batch_j_input[int(n_j[t])] - 2 * batch_j_input[int(g_j[t])]
                    if torch.tensor(10) < youxi_true < torch.tensor(40):
                        T+=1
                    youxi += abs(torch.tensor(25)-youxi_true)
                hetao.append(T)

                youxi = youxi/torch.tensor(70)
                youxi_num_batch.append(youxi)

            R = torch.tensor(youxi_num_batch)
            # print("平均奖励值",R.mean().item(),"平均合套率",(sum(hetao)/len(hetao))/50)
            # R += torch.norm(Y1-Y_ini, dim=1)  # 回到起点


            # self-critic base line
            mask = torch.zeros(B,size)
            
            C = 0
            baseline = 0
            
            Y = X_roll.view(B,size,1)
            x = Y[:,0,:]
            h = None
            c = None
            batch_list1 = []
            for k in range(size):
            
                output, h, c, _ = model(x=x, X_all=X_roll, h=h, c=c, mask=mask)
            
                # sampler = torch.distributions.Categorical(output)
                # idx = sampler.sample()         # now the idx has B elements
                idx = torch.argmax(output, dim=1)    # greedy baseline

                # 游隙计算
                batch_list1.extend(idx)
                # Y1 = Y[[i for i in range(B)], idx.data].clone()
                # if k == 0:
                #     Y_ini = Y1.clone()
                # if k > 0:
                #     baseline = torch.norm(Y1-Y0, dim=1)
                #
                # Y0 = Y1.clone()
                # x = Y[[i for i in range(B)], idx.data].clone()
                #
                # C += baseline
                mask[[i for i in range(B)], idx.data] += -np.inf

            youxi_num_batch1 = []
            for j in range(B):
                batch_reward_j = []
                for Q in range(size):  # size:150
                    batch_reward_j.append(batch_list1[64 * Q + j])
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
                for t in range(70):
                    HH = (batch_j_input[int(w_j[t])] - batch_j_input[int(n_j[t])] - 2 * batch_j_input[int(g_j[t])])
                    youxi += abs(torch.tensor(25) - HH)
                youxi = youxi / torch.tensor(70)
                youxi_num_batch1.append(youxi)

            C = torch.tensor(youxi_num_batch1)
        
            gap = (R-C).mean()
            loss = ((R-C-gap)*logprobs).mean()
        
            loss.backward()
            
            max_grad_norm = 1.0
            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                               max_grad_norm, norm_type=2)
            optimizer.step()
            opt_scheduler.step()
            print("epoch:{}, batch:{}/{}, reward:{}"
                  .format(epoch, i, steps, R.mean().item()))
            if i % 50 == 0:                                                             # 每50次进行测试
                # R_mean.append(R.mean().item())
                # R_std.append(R.std().item())
                
                # greedy validation                                                     # 贪婪验证
                
                tour_len = 0

                X = getset()
                X = torch.Tensor(X)
                
                mask = torch.zeros(B_val,size)
                
                R = 0
                logprobs = 0
                Idx = []
                reward = 0
                
                Y1 = X.view(B_val, size, 1)    # to the same batch size
                x = Y1[:,0,:]
                h = None
                c = None
                batch_list2 = []
                for k in range(size):
                    
                    output, h, c, hidden_u = model(x=x, X_all=X, h=h, c=c, mask=mask)
                    
                    sampler = torch.distributions.Categorical(output)
                    # idx = sampler.sample()
                    idx = torch.argmax(output, dim=1)
                    Idx.append(idx.data)
                
                    # Y1 = Y[[i for i in range(B_val)], idx.data]
                    #
                    # if k == 0:
                    #     Y_ini = Y1.clone()
                    # if k > 0:
                    #     reward = torch.norm(Y1-Y0, dim=1)
                    #
                    # Y0 = Y1.clone()
                    # x = Y[[i for i in range(B_val)], idx.data]
                    # R += reward

                    # 游隙计算
                    batch_list2.extend(idx)


                    mask[[i for i in range(B_val)], idx.data] += -np.inf

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
                    batch_j_input = Y1[j, :, :]
                    batch_j_input = batch_j_input.view(210)
                    youxi = 0
                    T = 0
                    for t in range(70):
                        youxi_true = (batch_j_input[int(w_j[t])] - batch_j_input[int(n_j[t])] - 2 * batch_j_input[int(g_j[t])])
                        youxi += abs(torch.tensor(25) - youxi_true)
                        if torch.tensor(10) < youxi_true < torch.tensor(40):
                            T+=1
                    hetao1.append(T)
                    youxi = youxi / torch.tensor(70)
                    youxi_num_batch2.append(youxi)

                R = torch.tensor(youxi_num_batch2)

                tour_len += R.mean().item()
                print('validation tour length:', tour_len,"合套率", (sum(hetao1)/len(hetao1))/70)

        print('save model to: ', save_root)
        torch.save(model, save_root)
