import gym
from stable_baselines.common.policies import MlpPolicy
from stable_baselines import TRPO
import pandas as pd
import time
import gym
from gym import spaces
from TradeEnv import TradeEnv
import numpy as np
from Util.Util import *

episode = 5000
EP_LEN = 300
FILE_TAG = "TRPO"
mode = "train"


def post_processor(state):
    return log10plus1R(state) / 10


env = TradeEnv(obs_time_size='60 day', obs_delta_frequency='1 day', sim_delta_time='1 day',
               start_episode=0, episode_len=EP_LEN, stock_code='000938_XSHE',
               result_path="E:/运行结果/TRPO/" + FILE_TAG + "/" + mode + "/", stock_data_path='../Data/'+mode+'/',
               poundage_rate=1.5e-3, reward_verbose=1, post_processor=post_processor,
               max_episode_days=EP_LEN)
env.seed(0)
env = env.unwrapped
model = TRPO(MlpPolicy, env, verbose=1, tensorboard_log="./log/")
model.learn(total_timesteps=episode * EP_LEN)
model.save("./model")


mode = 'test'
env = TradeEnv(obs_time_size='60 day', obs_delta_frequency='1 day', sim_delta_time='1 day',
               start_episode=0, episode_len=EP_LEN, stock_code='000938_XSHE',
               result_path= "E:/运行结果/TRPO/" + FILE_TAG + "/" + mode + "/", stock_data_path='../Data/train/',
               poundage_rate=1.5e-3, reward_verbose=1, post_processor=post_processor,
               max_episode_days=EP_LEN)
env.seed(0)
env = env.unwrapped
env.result_path = "E:/运行结果/TRPO/" + FILE_TAG + "/" + mode + "/"
for ep in range(10):
    print(ep)
    s = env.reset()
    for step in range(EP_LEN):
        a, _ = model.predict(s)
        s, _, done, _ = env.step(a)
        if done:
            break
    env.render("manual")
