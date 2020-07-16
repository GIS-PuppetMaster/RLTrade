import wandb
from Util.Util import *
from copy import deepcopy
import os
import tensorflow as tf

GPU = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = GPU
policy_args = dict(act_fun=gelu, net_arch=[dict(vf=[256, 128, 64, 32], pi=[256, 128, 64, 32])], l2_scale=0.,
                   dropout_rate=0.)
agent_state = False
episode = 100000
EP_LEN = 250 * 3
n_training_envs = 1
n_eval_episodes = 100
save_freq = EP_LEN * 50
eval_freq = EP_LEN * 50
seed = 0
stock_codes = ['000001_XSHE', '000002_XSHE', '000063_XSHE', '000069_XSHE', '000100_XSHE', '000157_XSHE', '000338_XSHE', '000413_XSHE', '000415_XSHE', '000423_XSHE', '000425_XSHE', '000538_XSHE', '000568_XSHE', '000596_XSHE', '000625_XSHE', '000627_XSHE', '000629_XSHE', '000630_XSHE', '000651_XSHE', '000656_XSHE', '000661_XSHE', '000671_XSHE', '000703_XSHE', '000709_XSHE', '000723_XSHE', '000725_XSHE', '000728_XSHE', '000768_XSHE', '000786_XSHE', '000858_XSHE', '000876_XSHE', '000895_XSHE', '000898_XSHE', '000938_XSHE', '000961_XSHE', '000963_XSHE', '002007_XSHE', '002008_XSHE', '002010_XSHE', '002024_XSHE', '002027_XSHE', '002032_XSHE', '002044_XSHE', '002050_XSHE', '002081_XSHE', '002120_XSHE', '002142_XSHE', '002146_XSHE', '002153_XSHE', '002179_XSHE', '002202_XSHE', '002230_XSHE', '002236_XSHE', '002241_XSHE', '002252_XSHE', '002271_XSHE', '002294_XSHE', '002304_XSHE', '002311_XSHE', '002415_XSHE', '300003_XSHE', '300015_XSHE', '300017_XSHE', '300024_XSHE', '300033_XSHE', '600000_XSHG', '600004_XSHG', '600009_XSHG', '600010_XSHG', '600011_XSHG', '600015_XSHG', '600016_XSHG', '600018_XSHG', '600019_XSHG', '600027_XSHG', '600028_XSHG', '600029_XSHG', '600030_XSHG', '600031_XSHG', '600036_XSHG', '600038_XSHG', '600048_XSHG', '600050_XSHG', '600061_XSHG', '600066_XSHG', '600068_XSHG', '600085_XSHG', '600089_XSHG', '600100_XSHG', '600104_XSHG', '600109_XSHG', '600111_XSHG', '600115_XSHG', '600118_XSHG', '600153_XSHG', '600170_XSHG', '600176_XSHG', '600177_XSHG', '600183_XSHG', '600188_XSHG', '600196_XSHG', '600208_XSHG', '600219_XSHG', '600221_XSHG', '600233_XSHG', '600271_XSHG', '600276_XSHG', '600297_XSHG', '600299_XSHG', '600309_XSHG', '600332_XSHG', '600340_XSHG', '600346_XSHG', '600352_XSHG', '600362_XSHG', '600369_XSHG', '600372_XSHG', '600383_XSHG', '600390_XSHG', '600398_XSHG', '600406_XSHG', '600436_XSHG', '600438_XSHG', '600482_XSHG', '600487_XSHG', '600489_XSHG', '600498_XSHG', '600516_XSHG', '600519_XSHG', '600522_XSHG', '600535_XSHG', '600547_XSHG', '600566_XSHG', '600570_XSHG', '600583_XSHG', '600585_XSHG', '600588_XSHG', '600606_XSHG', '600637_XSHG', '600655_XSHG', '600660_XSHG', '600663_XSHG', '600674_XSHG', '600690_XSHG', '600703_XSHG', '600741_XSHG', '600760_XSHG', '600795_XSHG', '600809_XSHG', '600816_XSHG', '600837_XSHG', '600848_XSHG', '600867_XSHG', '600886_XSHG', '600887_XSHG', '600893_XSHG', '600900_XSHG', '600999_XSHG', '601006_XSHG', '601009_XSHG', '601088_XSHG', '601166_XSHG', '601169_XSHG', '601186_XSHG', '601318_XSHG', '601398_XSHG', '601600_XSHG', '601601_XSHG', '601628_XSHG', '601766_XSHG', '601788_XSHG', '601898_XSHG', '601899_XSHG', '601989_XSHG', '601998_XSHG']
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
                        time_format="%Y-%m-%d", noise_rate=0.)
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
