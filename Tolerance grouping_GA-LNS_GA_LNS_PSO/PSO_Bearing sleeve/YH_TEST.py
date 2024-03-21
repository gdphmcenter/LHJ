
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei']

youxi = [5.2,4.9,4.8,3.8,3.8,3.8,3.8,3.6,4.9,4.9,4.9,6.0]
youxi.sort(reverse=True)
plt.plot([i for i in range(int(len(youxi)))], youxi, 'r-')
plt.ylabel("游隙指标")
plt.xlabel("迭代次数")
plt.legend(["PSO"])
plt.show()