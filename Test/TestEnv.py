import pickle as pk
from Util.Util import *
from Env.TradeEnv import TradeEnv
import numpy as np


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
                       stock_data_path='../Data/train/',
                       poundage_rate=2.5e-3, reward_verbose=1, post_processor=post_processor,
                       mode='eval', agent_state=True, feature_num=32, data_type='day',
                       time_format="%Y-%m-%d", noise_rate=0., load_from_cache=True)
env = TradeEnv(**eval_env_config)
obs = env.reset()
done = False
i=0
while not done:
    action_prefix = softmax(np.random.randn(len(stock_codes))+0.2).tolist()
    action = np.array(action_prefix + [np.random.random()])
    obs, reward, done, _ = env.step(action)
    i+=1
env.render('hybrid')
