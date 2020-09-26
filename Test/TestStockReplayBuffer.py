import json
import Tianshou
import torch
import pickle as pk
from Util.Util import *
from Env.TradeEnv import TradeEnv
import numpy as np
from Tianshou.StockReplayBuffer import *
from copy import copy, deepcopy


def softmax(x):
    x_exp = np.exp(x)
    # 如果是列向量，则axis=0
    x_sum = np.sum(x_exp, axis=0, keepdims=True)
    s = x_exp / x_sum
    return s


with open('../Data/000300_XSHG_list.pkl', 'rb') as f:
    stock_codes = pk.load(f)
exp_name = 'testEnv'
with open('../Config/TD3Config.json', 'r', encoding='utf-8') as f:
    config = json.load(f)
env = TradeEnv(**config['env']['test'], config=config)
obs = env.reset()
done = False
i = 0
buffer = StockPrioritizedReplayBuffer(**config['env']['test'], **config['train']['replay_buffer'])
local_buffer = []
for i in range(2):
    action_prefix = softmax(np.random.randn(config['env']['train']['trade_stock_num'])).tolist()
    action = np.array(action_prefix + [np.random.random()])
    obs_next, reward, done, info = env.step(action)
    local_buffer.append([deepcopy(obs), action, reward, done, deepcopy(obs_next), info])
    buffer.add(deepcopy(obs), action, reward, done, deepcopy(obs_next), info)
    obs = obs_next
sample = buffer.sample(2)
assert ((local_buffer[0][0]['stock_obs'] == sample[0]['obs']['stock_obs'][0]).all() and (
        local_buffer[1][0]['stock_obs'] == sample[0]['obs']['stock_obs'][1]).all()) or (
                   (local_buffer[0][0]['stock_obs'] == sample[0]['obs']['stock_obs'][1]).all() and (
                   local_buffer[1][0]['stock_obs'] == sample[0]['obs']['stock_obs'][0]).all())
