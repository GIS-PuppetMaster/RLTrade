from stable_baselines.common.policies import *
from stable_baselines import *
from stable_baselines.common.env_checker import check_env
from stable_baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from TradeEnv import TradeEnv
from Util.Util import *
from Util.Callback import CustomCallback
from stable_baselines.common.callbacks import CheckpointCallback
from stable_baselines.sac.policies import MlpPolicy as SACMlpPolicy
from stable_baselines.ddpg.policies import MlpPolicy as DDPGMlpPolicy
from stable_baselines.td3.policies import MlpPolicy as TD3MlpPolicy

episode = 2500
EP_LEN = 250 * 3
FILE_TAG = "TRPO"
mode = "train"
n_training_envs = 1


def post_processor(state):
    return log10plus1R(state) / 10


def make_env():
    env = TradeEnv(obs_time_size='60 day', obs_delta_frequency='1 day', sim_delta_time='1 day',
                   start_episode=0, episode_len=EP_LEN, stock_code='000938_XSHE',
                   result_path="E:/运行结果/TRPO/" + FILE_TAG + "/" + mode + "/",
                   stock_data_path='./Data/train/',
                   poundage_rate=1.5e-3, reward_verbose=1, post_processor=post_processor,
                   max_episode_days=EP_LEN)
    check_env(env)
    return env


env = DummyVecEnv([make_env for _ in range(n_training_envs)])
del_file('E:\运行结果\TRPO\TRPO/train')
monitorCallback = CustomCallback()
checkPointCallback = CheckpointCallback(save_freq=10 * EP_LEN, save_path='./checkpoints')
policy_args = dict(net_arch=[64, 128, dict(vf=[64], pi=[64])])
model = TRPO(MlpPolicy, env, verbose=1, tensorboard_log="./log/", seed=0)
model.learn(total_timesteps=episode * EP_LEN, callback=[monitorCallback, checkPointCallback])
model.save("./model")

model = TRPO.load('./model_old_stable.zip')
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
