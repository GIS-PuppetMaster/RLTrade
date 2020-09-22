import json
import pickle as pk
from time import time

from Tianshou.Net.NBeats import NBeatsNet
from Util.Util import *
from Env.TradeEnv import TradeEnv
import numpy as np
import torch


def softmax(x):
    x_exp = np.exp(x)
    # 如果是列向量，则axis=0
    x_sum = np.sum(x_exp, axis=0, keepdims=True)
    s = x_exp / x_sum
    return s


with open('../Data/000300_XSHG_list.pkl', 'rb') as f:
    stock_codes = pk.load(f)
exp_name = 'testEnv'
with open('../Config/TRPOConfigLinux.json', 'r', encoding='utf-8') as f:
    config = json.load(f)
time_steps, stocks, input_dim, output_dim = 60, 21, 32, 2
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Definition of the model.
# model = NBeatsNet(backcast_length=time_steps, forecast_length=output_dim,
#                   stack_types=(NBeatsNet.GENERIC_BLOCK, NBeatsNet.TREND_BLOCK, NBeatsNet.SEASONALITY_BLOCK),
#                   nb_blocks_per_stack=2,
#                   thetas_dims=(4, 4, 4), share_weights_in_stack=True, hidden_layer_units=64, device=device)
env = TradeEnv(**config['env']['test'], config=config)
env.seed(0)
times = 5
for _ in range(times):
    obs = env.reset()
    done = False
    i = 0
    t = time()
    while not done:
        action_prefix = softmax(np.random.randn(stocks)).tolist()
        action = np.array(action_prefix + [np.random.random()])
        obs, reward, done, _ = env.step(action)
        i += 1
# env.render('hybrid')
    print(time()-t)

