import json
import pickle as pk

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
with open('../Config/TestTD3Config.json', 'r', encoding='utf-8') as f:
    config = json.load(f)
time_steps, stocks, input_dim, output_dim = 60, 180, 32, 2
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Definition of the model.
model = NBeatsNet(backcast_length=time_steps, forecast_length=output_dim,
                  stack_types=(NBeatsNet.GENERIC_BLOCK, NBeatsNet.TREND_BLOCK, NBeatsNet.SEASONALITY_BLOCK),
                  nb_blocks_per_stack=2,
                  thetas_dims=(4, 4, 4), share_weights_in_stack=True, hidden_layer_units=64, device=device)
env = TradeEnv(**config['env']['test'], config=config)
env.seed(0)
obs = env.reset()
done = False
i = 0
while not done:
    res = model(torch.tensor(np.expand_dims(obs['stock_obs'], axis=0), device=device, dtype=torch.float))
    action_prefix = softmax(np.random.randn(len(stock_codes)) + 0.2).tolist()
    action = np.array(action_prefix + [np.random.random()])
    action += (env.action_space.low + env.action_space.high)/2
    obs, reward, done, _ = env.step(action)
    i += 1
env.render('native')
