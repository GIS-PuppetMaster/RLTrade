import wandb
from Util.Util import *
from copy import deepcopy
import os

GPU = "-1"
os.environ["CUDA_VISIBLE_DEVICES"] = GPU
policy_args = dict(act_fun=gelu, net_arch=[dict(vf=[256, 128, 64], pi=[256, 128, 64])], l2_scale=0.01)
agent_state = False
episode = 50000
EP_LEN = 250 * 3
n_training_envs = 1
agent_config = deepcopy(policy_args)
act = agent_config['act_fun']
agent_config['act_fun'] = act.__name__
agent_config['agent_state'] = agent_state
agent_config['episode'] = episode
agent_config['EP_LEN'] = EP_LEN
agent_config['n_training_envs'] = n_training_envs
agent_config['GPU'] = GPU
net_type = str(agent_config).replace(":", "-").replace("'", "").replace("{", "").replace("}", "").replace("[",
                                                                                                          "").replace(
    "]", "").replace(",", "_").replace(" ","")
train_env_config = dict(obs_time_size='60 day', obs_delta_frequency='1 day', sim_delta_time='1 day',
                        start_episode=0, episode_len=EP_LEN,
                        stock_codes=['000938_XSHE', '601318_XSHG', '601628_XSHG', '002049_XSHE',
                                     '000001_XSHE'],
                        result_path="E:/运行结果/TRPO/" + net_type + "/train/",
                        stock_data_path='./Data/train/',
                        poundage_rate=1.5e-3, reward_verbose=1, post_processor=post_processor, principal=1e6,
                        mode='train', agent_state=agent_state)
eval_env_config = dict(obs_time_size='60 day', obs_delta_frequency='1 day', sim_delta_time='1 day',
                       start_episode=0, episode_len=EP_LEN,
                       stock_codes=['000938_XSHE', '601318_XSHG', '601628_XSHG', '002049_XSHE',
                                    '000001_XSHE'],
                       result_path="E:/运行结果/TRPO/" + net_type + "/" + 'eval' + "/",
                       stock_data_path='./Data/test/',
                       poundage_rate=1.5e-3, reward_verbose=1, post_processor=post_processor, principal=1e6,
                       mode='eval', agent_state=agent_state, end_index_bound=-10)
train_env_config_ = deepcopy(train_env_config)["post_processor"] = post_processor.__name__
eval_env_config_ = deepcopy(eval_env_config)["post_processor"] = post_processor.__name__

config = dict(train_env_config=train_env_config_, eval_env_config=eval_env_config_, agent_config=agent_config)


def init_wandb():
    wandb.login()
    wandb.init(project='Stable-BaselineTrading', sync_tensorboard=True, config=config, name=net_type)
