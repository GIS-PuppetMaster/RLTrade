import pickle as pk
from Util.Util import *
from Env.TradeEnv import TradeEnv
import numpy as np
from Tianshou.StockReplayBuffer import StockReplayBuffer
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
eval_env_config = dict(obs_time_size=60, sim_delta_time=1,
                       start_episode=0, episode_len=750,
                       stock_codes=stock_codes,
                       result_path="E:/运行结果/TRPO/" + exp_name + "/" + 'eval' + "/",
                       stock_data_path='../Data/test/',
                       poundage_rate=2.5e-3, post_processor=[
        [
            "Util.Util",
            "post_processor"
        ],
        [
            "Util.Util",
            "log10plus1R"
        ],
        [
            "Util.Util",
            "log10plus1R"
        ]
    ],
                       agent_state=True, feature_num=32, data_type='day',
                       time_format="%Y-%m-%d", noise_rate=0., load_from_cache=True)
env = TradeEnv(**eval_env_config)
obs = env.reset()
done = False
i = 0
buffer = StockReplayBuffer(1000000, **eval_env_config)
local_buffer = []
for i in range(2):
    action_prefix = softmax(np.random.randn(len(stock_codes)) + 0.2).tolist()
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
