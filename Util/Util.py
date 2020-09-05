import os
import numpy as np
from sklearn.preprocessing import StandardScaler
import warnings
from typing import Dict, List, Union, Callable, Optional

warnings.filterwarnings('ignore')

scaler = StandardScaler()


def get_module(module: Optional[list]):
    import importlib
    if module is not None:
        module_name = module[0]
        module_ = importlib.import_module(module_name)
        # 递归导入
        for j in range(1, len(module)):
            module_ = module_.__dict__[module[j]]
    else:
        module_ = lambda x: x
    return module_


def get_modules(modules, index: Optional[int] = None):
    res_module = []
    if index is None:
        for module in modules:
            res_module.append(get_module(module))
    else:
        res_module = get_module(modules[index])
    return res_module


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


def post_processor(state):
    return scaler.fit_transform(state)


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
    import tensorflow as tf
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


def find_model(id, useVersion="final", main_path="./", timestamp=None):
    if id is None or id == "":
        raise ("id不能为空")
    fl = os.listdir('./wandb/')
    folder_name = None
    for file in fl:
        file_ele = file.split("-")
        ID = file_ele[-1]
        if id == ID:
            if timestamp is None or timestamp == "":
                folder_name = file
            elif file_ele[-2] == timestamp:
                folder_name = file
    if folder_name is None:
        raise ("未找到包含id:{}的文件夹".format(id))
    if useVersion == "last" or (useVersion == 'final' and not os.path.exists(
            os.path.join(main_path, 'wandb', folder_name, 'final_model.zip'))):
        model_path = os.path.join(main_path, 'wandb', folder_name, 'checkpoints/')
        file_list = os.listdir(model_path)
        max_index = -1
        max_file_name = ''
        for filename in file_list:
            index = int(filename.split("_")[2])
            if index > max_index:
                max_index = index
                max_file_name = filename
        model_path = os.path.join(model_path, max_file_name)
    elif useVersion == "final":
        model_path = os.path.join(main_path, 'wandb', folder_name, 'final_model.zip')
        max_file_name = "final"
    elif useVersion == "best":
        model_path = os.path.join(main_path, 'wandb', folder_name, 'best_model.zip')
        max_file_name = "best_model"
    else:
        model_path = os.path.join(main_path, 'wandb', folder_name, 'checkpoints', useVersion)
        max_file_name = useVersion
    return folder_name, model_path, max_file_name


def LoadCustomPolicyForTest(model_path):
    from stable_baselines import TRPO
    data, params = TRPO._load_from_file(model_path)
    # 设置dropout比率为0.，模型内部会自动设置training为False
    data['policy_kwargs']['dropout_rate'] = 0.
    model = TRPO(policy=data["policy"], env=None, _init_setup_model=False)
    model.__dict__.update(data)
    model.setup_model()
    model.load_parameters(params)
    return model
