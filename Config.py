import wandb
from Util.Util import *
from copy import deepcopy
import os
import tensorflow as tf

GPU = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = GPU
policy_args = dict(act_fun=gelu, net_arch=[dict(vf=[256, 128, 128, 32], pi=[256, 128,  128, 32])], l2_scale=0.1,
                   dropout_rate=0.)
agent_state = False
episode = 10000
EP_LEN = 250 * 3
n_training_envs = 1
n_eval_episodes = 50
save_freq = EP_LEN * 50
eval_freq = EP_LEN * 50
seed = 0
stock_codes = ["000938_XSHE", "002230_XSHE", "002415_XSHE", "000063_XSHE", "600460_XSHG", "002049_XSHE", "002371_XSHE",
               '000001_XSHE', '601318_XSHG', '601628_XSHG']
agent_config = deepcopy(policy_args)
agent_config['act_fun'] = agent_config['act_fun'].__name__
agent_config['net_arch'] = str(agent_config['net_arch']).replace("\"", "").replace("\'", "")
agent_config['agent_state'] = str(agent_state)
agent_config['episode'] = episode
agent_config['EP_LEN'] = EP_LEN
agent_config['n_training_envs'] = n_training_envs
agent_config['GPU'] = GPU
agent_config['save_freq'] = save_freq
agent_config['eval_freq'] = eval_freq
agent_config['n_eval_episodes'] = n_eval_episodes
agent_config['seed'] = seed
exp_name = (str(agent_config) + "") \
    .replace(":", "-") \
    .replace("'", "") \
    .replace("{", "") \
    .replace("}", "") \
    .replace("[", "") \
    .replace("]", "") \
    .replace(",", "_") \
    .replace(" ", "")
train_env_config = dict(obs_time_size=60, obs_delta_frequency=1, sim_delta_time=1,
                        start_episode=0, episode_len=EP_LEN,
                        stock_codes=stock_codes,
                        result_path="E:/运行结果/TRPO/" + exp_name + "/train/",
                        stock_data_path='./Data/train/',
                        poundage_rate=1.5e-3, reward_verbose=1, post_processor=post_processor,
                        mode='train', agent_state=agent_state, feature_num=26, data_type='day',
                        time_format="%Y-%m-%d", noise_rate=1)
eval_env_config = dict(obs_time_size=60, obs_delta_frequency=1, sim_delta_time=1,
                       start_episode=0, episode_len=250,
                       stock_codes=stock_codes,
                       result_path="E:/运行结果/TRPO/" + exp_name + "/" + 'eval' + "/",
                       stock_data_path='./Data/test/',
                       poundage_rate=1.5e-3, reward_verbose=1, post_processor=post_processor,
                       mode='eval', agent_state=agent_state, feature_num=26, data_type='day',
                       time_format="%Y-%m-%d", noise_rate=0.)
train_env_config_ = deepcopy(train_env_config)
train_env_config_["post_processor"] = post_processor.__name__
eval_env_config_ = deepcopy(eval_env_config)
eval_env_config_["post_processor"] = post_processor.__name__

config = dict(train_env_config=train_env_config_, eval_env_config=eval_env_config_, agent_config=agent_config)


def init_wandb():
    wandb.init(project='Stable-BaselineTrading', sync_tensorboard=True, config=config, name=exp_name)
