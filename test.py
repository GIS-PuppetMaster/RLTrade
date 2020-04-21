from stable_baselines.common.policies import *
from stable_baselines import *
from TradeEnv import TradeEnv
from Util.Util import *
import os

episode = 2500
EP_LEN = 250 * 3
FILE_TAG = "TRPO"
mode = "train"
n_training_envs = 1


def post_processor(state):
    return log10plus1R(state) / 10



file_list = os.listdir('./checkpoints/')
max_index = -1
max_file_name = ''
for filename in file_list:
    index = int(filename.split("_")[2])
    if index > max_index:
        max_index = index
        max_file_name = filename
max_file_name = 'rl_model_659456_steps.zip'
model = TRPO.load('./checkpoints/' + max_file_name)
mode = 'test'
env = TradeEnv(obs_time_size='60 day', obs_delta_frequency='1 day', sim_delta_time='1 day',
               start_episode=0, episode_len=EP_LEN, stock_code='000938_XSHE',
               result_path="E:/运行结果/TRPO/" + FILE_TAG + "/" + mode + "/",
               stock_data_path='./Data/test/',
               poundage_rate=1.5e-3, reward_verbose=1, post_processor=post_processor,
               max_episode_days=EP_LEN)
env.seed(0)
env = env.unwrapped
env.result_path = "E:/运行结果/TRPO/" + FILE_TAG + "/" + mode + "/"
for ep in range(50):
    print(ep)
    s = env.reset()
    for step in range(EP_LEN):
        a, _ = model.predict(s)
        s, _, done, _ = env.step(a)
        if done:
            break
    env.render("manual")
