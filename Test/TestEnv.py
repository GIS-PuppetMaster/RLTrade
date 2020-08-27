import json
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
with open('../Config/TestTD3Config.json', 'r', encoding='utf-8') as f:
    config = json.load(f)
env = TradeEnv(**config['env']['test'], config=config)
obs = env.reset()
done = False
i = 0
while not done:
    action_prefix = softmax(np.random.randn(len(stock_codes)) + 0.2).tolist()
    action = np.array(action_prefix + [np.random.random()])
    obs, reward, done, _ = env.step(action)
    i += 1
env.render('hybrid')
