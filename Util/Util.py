import os
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

scaler = StandardScaler()


def log2plus1R(x):
    x = np.sign(x) * np.log2(np.abs(x + np.sign(x)))
    x[np.isnan(x)] = 0
    x[np.isinf(x)] = 0
    return x


def log10plus1R(x):
    x = np.sign(x) * np.log10(np.abs(x + np.sign(x)))
    x[np.isnan(x)] = 0
    x[np.isinf(x)] = 0
    return x


def post_processor(state, agent_state):
    if agent_state:
        price = state[0:-2]
        s = state[-2:]
        price = price.reshape(60, 26)
        price = scaler.fit_transform(price)
        s = log10plus1R(s) / 10
        # return log10plus1R(state) / 10
        return np.append(price.flatten(), s)
    else:
        price = state.reshape(60, 26)
        return scaler.fit_transform(price).flatten()



def del_file(path_data):
    try:
        for i in os.listdir(path_data):  # os.listdir(path_data)#返回一个列表，里面是当前目录下面的所有东西的相对路径
            file_data = path_data + "\\" + i  # 当前文件夹的下面的所有东西的绝对路径
            if os.path.isfile(file_data) == True:  # os.path.isfile判断是否为文件,如果是文件,就删除.如果是文件夹.递归给del_file.
                os.remove(file_data)
            else:
                del_file(file_data)
        return True
    except:
        return False


def gelu(input_tensor):
    cdf = 0.5 * (1.0 + tf.erf(input_tensor / tf.sqrt(2.0)))
    return input_tensor * cdf


def fill_inf(array):
    for i in range(array.shape[1]):  # 遍历每一列（每一列中的nan替换成该列的均值）
        # 跳过日期
        if i == 0:
            continue
        temp_col = array[:, i]  # 当前的一列
        mean = np.mean(temp_col[~np.isinf(temp_col.astype(np.float))])
        for j in range(array.shape[0]):
            value = float(temp_col[j])
            if np.isinf(value):
                array[j, i] = mean
    return array




