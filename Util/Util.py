import os
from collections import OrderedDict
from copy import deepcopy
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import warnings
from typing import Dict, List, Union, Callable, Optional
import cupy as cp
from numba import jit
import dill
warnings.filterwarnings('ignore')

scaler = StandardScaler()
force_apply_in_step = ['norm_processor']

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


def covert_type(x):
    if not isinstance(x, cp.ndarray) and np.prod(x.shape) > 100000:
        x = cp.asarray(x)
        import cupy as F
    else:
        import numpy as F
    return x, F

@jit
def log2plus1R(x):
    x, F = covert_type(x)
    sign = F.sign(x)
    x = sign * F.log2(F.abs(x + sign))
    x[F.isinf(x)] = 0.
    x[F.isnan(x)] = 0.
    return cp.asnumpy(x) if isinstance(x, cp.ndarray) else x

@jit
def log10plus1R(x):
    x, F = covert_type(x)
    sign = F.sign(x)
    x = sign * F.log10(F.abs(x + sign))
    x[F.isinf(x)] = 0.
    x[F.isnan(x)] = 0.
    return cp.asnumpy(x) if isinstance(x, cp.ndarray) else x


def norm_processor(state):
    state_shape = state.shape
    return scaler.fit_transform(state.reshape(state.shape[0], -1)).reshape(state_shape)

@jit
def selective_log10plus1R(x):
    x, F = covert_type(x)
    x[F.abs(x) > 10] = log10plus1R(x[F.abs(x) > 10])
    return cp.asnumpy(x) if isinstance(x, cp.ndarray) else x

@jit
def diff_log10plus1R(x):
    x, F = covert_type(x)
    x = F.diff(x, prepend=0)
    return log10plus1R(x)

@jit
def diff_selective_log10plus1R(x):
    x, F = covert_type(x)
    x = F.diff(x, prepend=0)
    return selective_log10plus1R(x)


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

def read_stock_data(stock_data_path, stock_codes, post_processor, data_type, feature_num, load_from_cache=False, **kwargs):
    save_path = os.path.join(stock_data_path, 'TradeEnvData.dill')
    # in order by stock_code
    stock_codes = [stock_code.replace('.', '_') for stock_code in stock_codes]
    stock_codes = sorted(stock_codes)
    miss_match = False
    if load_from_cache and os.path.exists(save_path):
        try:
            with open(save_path, 'rb') as f:
                stock_codes_, time_series, global_date_intersection = dill.load(f)
            miss_match = stock_codes != stock_codes_ or list(time_series.values())[0].shape[1] != feature_num
            if not miss_match:
                print("数据读取完毕")
            else:
                print("数据不匹配，重新读入并覆盖")
        except:
            miss_match = True

    if not load_from_cache or not os.path.exists(save_path) or miss_match:
        stocks = OrderedDict()
        date_index = []
        for idx, stock_code in enumerate(stock_codes):
            print(f'{idx + 1}/{len(stock_codes)} loaded:{stock_code}')
            if data_type == 'day':
                raw = pd.read_csv(stock_data_path + stock_code + '_with_indicator.csv', index_col=False)
            else:
                raise Exception(f"Wrong data type for:{data_type}")
            raw_moneyflow = pd.read_csv(stock_data_path + stock_code + '_moneyflow.csv', index_col=False)[
                ['date', 'change_pct', 'net_pct_main', 'net_pct_xl', 'net_pct_l', 'net_pct_m', 'net_pct_s']].apply(
                lambda x: x / 100 if isinstance(x[1], np.float64) else x)
            raw = pd.merge(raw, raw_moneyflow, left_on='Unnamed: 0', right_on='date', sort=False, copy=False).drop(
                'date', 1).rename(columns={'Unnamed: 0': 'date'})
            raw.fillna(method='ffill', inplace=True)
            date_index.append(np.array(raw['date']))
            raw.set_index('date', inplace=True)
            stocks[stock_code] = raw
        # 生成各支股票数据按时间的并集
        global_date_intersection = date_index[0]
        for i in range(1, len(date_index)):
            global_date_intersection = np.union1d(global_date_intersection, date_index[i])
        global_date_intersection = global_date_intersection.tolist()
        # 根据并集补全停牌数据
        for key in stocks.keys():
            value = stocks[key]
            date = value.index.tolist()
            # 需要填充的日期
            fill = np.setdiff1d(global_date_intersection, date)
            for fill_date in fill:
                value.loc[fill_date] = [np.nan] * value.shape[1]
            if len(fill) > 0:
                value.sort_index(inplace=True)
            stocks[key] = np.array(value)
        # 生成股票数据
        time_series = OrderedDict()
        # in order by stock_codes
        # 当前时间在state最后
        for i in range(len(global_date_intersection)):
            date = global_date_intersection[i]
            stock_data_in_date = []
            for key in stocks.keys():
                value = stocks[key][i, :]
                stock_data_in_date.append(value.tolist())
            time_series[date] = np.array(stock_data_in_date)
        with open(save_path, 'wb') as f:
            dill.dump((stock_codes, time_series, global_date_intersection), f)
    time_series_without_nan = deepcopy(time_series)
    if post_processor[0].__name__ in force_apply_in_step:
        F = lambda x: x
    else:
        F = lambda x: post_processor[0](x)
    time_series_without_nan.update(map(lambda t: (t[0], F(np.nan_to_num(t[1], nan=0., posinf=0., neginf=0.))), time_series_without_nan.items()))
    return stock_codes, time_series, time_series_without_nan, global_date_intersection

