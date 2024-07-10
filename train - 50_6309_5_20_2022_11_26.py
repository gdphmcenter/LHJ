
import argparse
import numpy as np
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from tqdm import tqdm
from gpn import GPN
import pandas as pd
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
from scipy.integrate import cumtrapz
import matplotlib.pyplot as plt
import random


def getset():
    data_set = []
    zuhe = []
    for l in range(64):
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

                # 读取内沟径值数据并拟合KDE
                x_train_inner = read_excel_data(file_path, inner_column_name)
                kde_inner = KernelDensity()
                grid_inner = GridSearchCV(kde_inner,
                                          {'bandwidth': h_vals, 'kernel': kernels},
                                          scoring=my_scores,
                                          cv=5)  # 使用5折交叉验证
                grid_inner.fit(x_train_inner)
                best_kde_inner = grid_inner.best_estimator_

                # 读取外沟尺寸数据并拟合KDE
                x_train_outer = read_excel_data(file_path, outer_column_name)
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
                        result_x_inner.append(x_test_inner[0, 0])
                    elif idx >= len(cdf_inner) - 1:
                        result_x_inner.append(x_test_inner[-1, 0])
                    else:
                        result_x_inner.append(x_test_inner[idx, 0])
                        # 查找外沟尺寸对应的横坐标值
                result_x_outer = []
                for prob in target_probs:
                    idx = np.searchsorted(cdf_outer, prob, side='right') - 1
                    if idx < 0:
                        result_x_outer.append(x_test_outer[0, 0])
                    elif idx >= len(cdf_outer) - 1:
                        result_x_outer.append(x_test_outer[-1, 0])
                    else:
                        result_x_outer.append(x_test_outer[idx, 0])

                zuhe.extend(result_x_outer)
                zuhe.extend(result_x_inner)
            # 产生滚动体
            if j == 1:
                job_g_piancha = list(np.random.choice([-8, -6, -4, -2, 0, 2, 4], size=50, replace=True))
                zuhe.extend(job_g_piancha)
    zuhe = np.array(zuhe)
    zuhe = zuhe.reshape(64,150,1)          # batch、size、value
    data_set = zuhe
    # print(data_set)
    return data_set

if __name__ == "__main__":
    cun_list = []
    # args
    parser = argparse.ArgumentParser(description="GPN with RL")
    parser.add_argument('--size', default=150, help="size of TSP")
    parser.add_argument('--epoch', default=2, help="number of epochs")
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
    save_root ='./model/gpn_tsp'+str(size)+"random_sample_input"+'.pt'
    
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
    # X_val_roll = getset()

    C = 0     # baseline
    R = 0     # reward

    # R_mean = []
    # R_std = []
    hetaolv = []
    R_mean = []
    plt_reward_training = []
    for epoch in range(n_epoch):
        for i in tqdm(range(steps)):
            optimizer.zero_grad()

            X_roll = getset()
            cun_list.append(X_roll)
            X_roll = torch.Tensor(X_roll)
            mask = torch.zeros(B,150)
        
            R = 0
            logprobs = 0
            reward = 0
            
            Y = X_roll.view(B,150,1)

            x = Y[:, 0, :]
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
                for t in range(50):
                    youxi_true = batch_j_input[int(w_j[t])] - batch_j_input[int(n_j[t])] - 2 * batch_j_input[int(g_j[t])]
                    if torch.tensor(5) < youxi_true < torch.tensor(20):
                        T+=1
                    youxi += abs(torch.tensor(12.5)-youxi_true)
                hetao.append(T/50)

                youxi = youxi/torch.tensor(50)
                youxi_num_batch.append(youxi)
            hetaolv.append(np.mean(hetao))

            R = torch.tensor(youxi_num_batch)
            # print("平均奖励值",R.mean().item(),"平均合套率",(sum(hetao)/len(hetao))/50)
            # R += torch.norm(Y1-Y_ini, dim=1)  # 回到起点
            R_mean.append(R.mean())

            # self-critic base line
            mask = torch.zeros(B,size)
            
            C = 0
            baseline = 0
            
            Y = X_roll.view(B,size,1)
            x = Y[:, 0, :]
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
                    if r < torch.tensor(50):
                        w_j.append(r)
                    if torch.tensor(49) < r < torch.tensor(100):
                        n_j.append(r)
                    if r > torch.tensor(99):
                        g_j.append(r)
                batch_j_input = Y[j, :, :]
                batch_j_input = batch_j_input.view(150)
                youxi = 0
                for t in range(50):
                    HH = (batch_j_input[int(w_j[t])] - batch_j_input[int(n_j[t])] - 2 * batch_j_input[int(g_j[t])])
                    youxi += abs(torch.tensor(12.5) - HH)
                youxi = youxi / torch.tensor(50)
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
            plt_reward_training.append(R.mean())
            print("epoch:{}, batch:{}/{}, reward:{}"
                  .format(epoch, i, steps, R.mean().item()))
            if i % 10 == 0:                                                             # 每50次进行测试
                # R_mean.append(R.mean().item())
                # R_std.append(R.std().item())
                
                # greedy validation                                                     # 贪婪验证
                
                tour_len = 0

                X = getset()
                # print(X)
                cun_list.append(X)
                X = torch.Tensor(X)
                
                mask = torch.zeros(B_val,size)
                
                R = 0
                logprobs = 0
                Idx = []
                reward = 0
                
                Y1 = X.view(B_val, size, 1)    # to the same batch size
                x = Y1[:, 0, :]
                h = None
                c = None
                batch_list2 = []
                for k in range(size):

                    output, h, c, hidden_u = model(x=x, X_all=X, h=h, c=c, mask=mask)
                    
                    sampler = torch.distributions.Categorical(output)
                    # idx = sampler.sample()
                    idx = torch.argmax(output, dim=1)
                    Idx.append(idx.data)


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
                        if r < torch.tensor(50):
                            w_j.append(r)
                        if torch.tensor(49) < r < torch.tensor(100):
                            n_j.append(r)
                        if r > torch.tensor(99):
                            g_j.append(r)
                    batch_j_input = Y1[j, :, :]
                    batch_j_input = batch_j_input.view(150)
                    youxi = 0
                    T = 0
                    for t in range(50):
                        youxi_true = (batch_j_input[int(w_j[t])] - batch_j_input[int(n_j[t])] - 2 * batch_j_input[int(g_j[t])])
                        youxi += abs(torch.tensor(12.5) - youxi_true)
                        if torch.tensor(5) < youxi_true < torch.tensor(20):
                            T+=1
                    hetao1.append(T)
                    youxi = youxi / torch.tensor(50)
                    youxi_num_batch2.append(youxi)

                R = torch.tensor(youxi_num_batch2)

                tour_len += R.mean().item()
                print('validation tour length:', tour_len,"合套率", (sum(hetao1)/len(hetao1))/50)
        plt.rcParams['font.sans-serif']=['SimHei']
        plt.rcParams['axes.unicode_minus']=False
        plt.plot([i for i in range(steps)], R_mean, ls='-', c="b")
        plt.legend()
        plt.xlabel('step')
        plt.ylabel('avg_value')
        plt.title(u'training中实际游隙与最佳游隙差值的平方均值')
        plt.show()
        plt.plot([i for i in range(steps)], hetaolv, ls='-', c="b")
        plt.legend()
        plt.xlabel('step')
        plt.ylabel('avg_value')
        plt.title(u'training中合套率的均值')
        plt.show()
        print('save model to: ', save_root)
        torch.save(model, save_root)
