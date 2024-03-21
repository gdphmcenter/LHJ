import numpy as np
np.random.seed(4)
w = np.random.normal(loc=22.5, scale=7.5, size=70)
n = np.random.normal(loc=-24, scale=2, size=70)
g = np.random.normal(loc=10.75, scale=2.75, size=70)

import pandas as pd
df = pd.DataFrame({'外圈': w,"内圈":n,"滚动体":g})
df.to_excel("G:/graph-pointer-network-master/整数规划模型/正态分布-70规模/random_4_70_new_input.xlsx", index=False)