
import gurobipy as gp
from gurobipy import GRB
from gurobipy import LinExpr
import pandas as pd
import numpy as np
import time
import re

# Prepare model parameters
# 读取外圈、内圈、滚动体的尺寸
df = pd.read_excel("random_4_70_new_input.xlsx")
data = df.values
X = data[:,0]
Y = data[:,1]
Z = data[:,2]
# 设置其他参数
Pmin = 10
Pmax = 40
Pbest = 25
N = len(X)
a1 = 100       # 超出游隙范围的惩罚系数
a2 = 0.1/N     # 实际游隙与最佳游隙差值平法的惩罚系统
# hard constraint model##############
# startT = time.perf_counter()
# BigM = 10000
# m = gp. Model (" bearing ")
# c = m.addVars(N,N,N,vtype = GRB.BINARY,name ="c")
# objExpr2 = LinExpr()
# for i in range(N):
#     for j in range(N):
#         for k in range(N):
#             objExpr2 += c[i,j,k] * (X[i] - Y[j] - 2*Z[k] - Pbest) * (X[i] - Y[j] - 2*Z[k] - Pbest)
# m.setObjective (objExpr2, GRB. MINIMIZE)
# # Add constraint
# for i in range(N):
#     cons = 0
#     for j in range(N):
#         for k in range(N):
#             cons += c[i,j,k]
#     m.addConstr (cons == 1)
# for j in range(N):
#     cons = 0
#     for i in range(N):
#         for k in range(N):
#             cons += c[i,j,k]
#     m.addConstr (cons == 1)
# for k in range(N):
#     cons = 0
#     for i in range(N):
#         for j in range(N):
#             cons += c[i,j,k]
#     m.addConstr (cons == 1)
# for i in range(N):
#     for j in range(N):
#         for k in range(N):
#             m.addConstr(X[i]-Y[j]-2*Z[k]+(1-c[i,j,k])*BigM >= Pmin)
#             m.addConstr(X[i] - Y[j] - 2 * Z[k] + (c[i, j, k] - 1) * BigM <= Pmax)
# m. optimize ()
# status = m.status
# endT = time.perf_counter()
# runTime = endT - startT
# v_c = m.getVars()
# # 读出决策变量的值，保存合套方案
# planHT = np.zeros((N,3))
# showC = list()
# rowIndex = 0
# youxicha = 0.0 #保存平均游隙差值（平方）
# hetao = 0 #保存合套率
# for varV in v_c:
#     if varV.X == 1.0 and varV.VarName.split('[',1)[0] == 'c':
#         showC.append(varV.VarName)
#         indexList = varV.VarName.split('[',1)[1].split(']',1)[0].split(',')
#         for columIndex in range(len(indexList)):
#             planHT[rowIndex,columIndex] = data[int(indexList[columIndex]),columIndex]
#         youxicha = youxicha + (planHT[rowIndex,0]-planHT[rowIndex,1]-2*planHT[rowIndex,2]-Pbest)*(planHT[rowIndex,0]-planHT[rowIndex,1]-2*planHT[rowIndex,2]-Pbest)
#         if (planHT[rowIndex,0]-planHT[rowIndex,1]-2*planHT[rowIndex,2]) >= Pmin and (planHT[rowIndex,0]-planHT[rowIndex,1]-2*planHT[rowIndex,2]) <= Pmax:
#             hetao = hetao + 1
#         rowIndex = rowIndex+1
# dfout = pd.DataFrame(data=planHT[0:,0:],columns=['外圈','内圈','滚动体'])
# #计算平均游隙差值（平方）和合套率
# youxicha = youxicha/N
# hetao = 100*hetao/N
# str = "Results_hard_" + str(N) + "N_" + str(round(runTime,2)) +"s_" + str(round(hetao,2)) + "%_" + str(round(youxicha,2))+"random_1_300-hard" + ".xlsx"
# dfout.to_excel(str)
# # hard constraint model##############

# soft constraint model##############
# Create a new model
startT = time.perf_counter()
m = gp. Model (" bearing ")

# Create variables
c = m.addVars(N,N,N,vtype = GRB.BINARY,name ="c")
d1_plus = m.addVars(N,N,N,lb=0.0, vtype = GRB.CONTINUOUS,name ="d1_plus")
d1_mins = m.addVars(N,N,N,lb=0.0, vtype = GRB.CONTINUOUS,name ="d1_mins")
d2_plus = m.addVars(N,N,N,lb=0.0, vtype = GRB.CONTINUOUS,name ="d2_plus")
d2_mins = m.addVars(N,N,N,lb=0.0, vtype = GRB.CONTINUOUS,name ="d2_mins")
# Set objective
objExpr1 = 0
objExpr2 = 0
for i in range(N):
    for j in range(N):
        for k in range(N):
            objExpr1 += c[i,j,k] * (d1_mins[i,j,k] + d2_plus[i,j,k])
            objExpr2 += c[i,j,k] * (X[i] - Y[j] - 2*Z[k] - Pbest) * (X[i] - Y[j] - 2*Z[k] - Pbest)
m.setObjective (a1*objExpr1 + a2*objExpr2, GRB. MINIMIZE)
# Add constraint
for i in range(N):
    cons = 0
    for j in range(N):
        for k in range(N):
            cons += c[i,j,k]
    m.addConstr (cons == 1)

for j in range(N):
    cons = 0
    for i in range(N):
        for k in range(N):
            cons += c[i,j,k]
    m.addConstr (cons == 1)

for k in range(N):
    cons = 0
    for i in range(N):
        for j in range(N):
            cons += c[i,j,k]
    m.addConstr (cons == 1)

for i in range(N):
    for j in range(N):
        for k in range(N):
            m.addConstr (c[i,j,k] * (X[i]-Y[j]-2*Z[k]+d1_mins[i,j,k]-d1_plus[i,j,k]-Pmin) == 0)
            m.addConstr (c[i,j,k] * (X[i]-Y[j]-2*Z[k]+d2_mins[i,j,k]-d2_plus[i,j,k]-Pmax) == 0)

# Optimize model
m. optimize ()
status = m.status
endT = time.perf_counter()
runTime = endT - startT
v_c = m.getVars()
# 读出决策变量的值，保存合套方案
planHT = np.zeros((N,3))
showC = list()
rowIndex = 0
youxicha = 0.0 #保存平均游隙差值（平方）
hetao = 0 #保存合套率
for varV in v_c:
    if varV.X == 1.0 and varV.VarName.split('[',1)[0] == 'c':
        showC.append(varV.VarName)

        indexList = varV.VarName.split('[',1)[1].split(']',1)[0].split(',')
        for columIndex in range(len(indexList)):
            planHT[rowIndex,columIndex] = data[int(indexList[columIndex]),columIndex]
        youxicha = youxicha + (planHT[rowIndex, 0] - planHT[rowIndex, 1] - 2 * planHT[rowIndex, 2] - Pbest) * (
                    planHT[rowIndex, 0] - planHT[rowIndex, 1] - 2 * planHT[rowIndex, 2] - Pbest)
        if (planHT[rowIndex, 0] - planHT[rowIndex, 1] - 2 * planHT[rowIndex, 2]) >= Pmin and (
                planHT[rowIndex, 0] - planHT[rowIndex, 1] - 2 * planHT[rowIndex, 2]) <= Pmax:
            hetao = hetao + 1
        rowIndex = rowIndex+1
dfout = pd.DataFrame(data=planHT[0:,0:],columns=['外圈','内圈','滚动体'])
#计算平均游隙差值（平方）和合套率
youxicha = youxicha/N
hetao = 100*hetao/N
str = "Results_soft_" + str(N) + "N_" + str(round(runTime,2)) +"s_" + str(round(hetao,2)) + "%_" + str(round(youxicha,2)) +"random_4_70_new_input.xlsx"+ ".xlsx"
dfout.to_excel(str)
# soft constraint model##############


# # Create a new model
# m = gp. Model (" mip1 ")
# # Create variables
# x = m. addVar ( vtype =GRB.BINARY , name ="x")
# y = m. addVar ( vtype =GRB.BINARY , name ="y")
# z = m. addVar ( vtype =GRB.BINARY , name ="z")
# # Set objective
# m. setObjective (x + y + 2 * z, GRB. MAXIMIZE )
# # Add constraint : x + 2 y + 3 z <= 4
# m. addConstr (x + 2 * y + 3 * z <= 4, "c0")
# # Add constraint : x + y >= 1
# m. addConstr (x + y >= 1, "c1")
# # Optimize model
# m. optimize ()
# for v in m.getVars():
#     print(v.x)
# print(m.ObjVal)