import os
import tensorflow.compat.v1 as tf
import numpy as np


def log2plus1R(x):
    return np.sign(x) * np.log2(np.abs(x + 1))


def log10plus1R(x):
    return np.sign(x) * np.log10(np.abs(x + 1))


def del_file(path_data):
    try:
        for i in os.listdir(path_data):  # os.listdir(path_data)#返回一个列表，里面是当前目录下面的所有东西的相对路径
            file_data = path_data + "\\" + i  # 当前文件夹的下面的所有东西的绝对路径
            if os.path.isfile(file_data) == True:  # os.path.isfile判断是否为文件,如果是文件,就删除.如果是文件夹.递归给del_file.
                os.remove(file_data)
            else:
                del_file(file_data)
    except:
        pass


def gelu(input_tensor):
    cdf = 0.5 * (1.0 + tf.erf(input_tensor / tf.sqrt(2.0)))
    return input_tensor * cdf
