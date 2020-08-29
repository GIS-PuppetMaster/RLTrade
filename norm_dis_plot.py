import pandas as pd  # 导入另一个包“pandas” 命名为 pd，理解成pandas是在 numpy 基础上的升级包
import numpy as np  # 导入一个数据分析用的包“numpy” 命名为 np
import matplotlib.pyplot as plt
from tianshou.exploration.random import OUNoise


# 正态分布的概率密度函数。可以理解成 x 是 mu（均值）和 sigma（标准差）的函数
def normfun(x, mu, sigma):
    pdf = np.exp(-((x - mu) ** 2) / (2 * sigma ** 2)) / (sigma * np.sqrt(2 * np.pi))
    return pdf


mean = 0
std = 0.1
theta = 0.5
dt=1e-2
size = 500
# 设定 x 轴前两个数字是 X 轴的开始和结束，第三个数字表示步长，或者区间的间隔长度
x = np.arange(-1, 1, 1/size*2)
# 设定 y 轴，载入刚才的正态分布函数
y = normfun(x, mean, std)
# noise = OUNoise(mu=mean, sigma=std, theta=theta, dt=dt)
# y = noise((size,))
plt.plot(x, y)

plt.title('Time distribution')
plt.xlabel('Time')
plt.ylabel('Probability')
# 输出
plt.show()
