import json
import wandb
import sys

if sys.platform == 'win32':
    sys.path.append("D:\\PycharmProjects\\Stable-BaselineTrading\\")
else:
    sys.path.append("/usr/zkx/Stable-BaselineTrading/")
from stable_baselines import TRPO
from stable_baselines.common.env_checker import check_env
from Env.RollTradeEnv import TradeEnv

from Util.Callback import CustomCallback
from Util.BestModelCallback import *
from stable_baselines.common.callbacks import *
from stable_baselines.common.vec_env import *
import shutil
from Util.CustomPolicy import CustomMultiStockPolicy
import argparse
from stable_baselines.common.callbacks import *


def make_env(config, seed, mode, id):
    env_config = config['env']['common']

    def get_env():
        env = TradeEnv(config=config, env_id=id, **env_config)
        env.seed(seed)
        check_env(env)
        return env

    return get_env


def replace_path(path, insert_fold):
    if path[-1] == '/':
        path = path[:-1]
    tup = os.path.split(path)
    return os.path.join(tup[0], insert_fold, tup[1], "", )


if __name__ == '__main__':
    name = input("please input run_name:\n")
    # name = 'debug'
    argparse = argparse.ArgumentParser()
    argparse.add_argument('--config', type=str, default=None)
    args = argparse.parse_args()
    with open(args.config, 'r', encoding='utf-8') as f:
        conf = json.load(f)
    conf['env']['common']['result_path'] = replace_path(conf['env']['common']['result_path'], name)
    if conf['global_wandb']:
        wandb.init(project='Stable-BaselineTradingV4', sync_tensorboard=True, config=conf, tensorboard=True, notes=name)

    os.environ["CUDA_VISIBLE_DEVICES"] = conf['train']['gpu']
    env = DummyVecEnv([make_env(conf, conf['seed'], 'train', id) for id in range(conf['env']['train_env_num'])])
    eval_env = TradeEnv(config=conf, **conf['env']['common'], env_id=0)
    eval_env.env_type = 'test'
    monitorCallback = CustomCallback()
    checkPointCallback = CheckpointCallback(save_freq=conf['train']['save_freq'], save_path=os.path.join(wandb.run.dir, conf['train']['save_dir']))
    last_mean_reward = -np.inf
    best_mean_reward = -np.inf
    idx = 0
    while True:
        print(f"Round:{idx}")
        model = TRPO(CustomMultiStockPolicy, env, tensorboard_log=conf['train']['log_dir'], seed=conf['seed'], **conf['policy'])
        model.learn(total_timesteps=conf['train']['total_timesteps'])
        split_point = env.envs[0].split_point
        eval_env.split_point = split_point
        episode_rewards, episode_lengths = evaluate_policy(model, eval_env,
                                                           n_eval_episodes=1,
                                                           render=conf['eval']['render'],
                                                           deterministic=conf['eval']['deterministic'],
                                                           return_episode_rewards=True)
        mean_reward, std_reward = np.mean(episode_rewards), np.std(episode_rewards)
        mean_ep_length, std_ep_length = np.mean(episode_lengths), np.std(episode_lengths)
        # Keep track of the last evaluation, useful for classes that derive from this callback
        last_mean_reward = mean_reward
        print("Eval "
              "episode_reward={:.2f} +/- {:.2f}".format(mean_reward, std_reward))
        print("Episode length: {:.2f} +/- {:.2f}".format(mean_ep_length, std_ep_length))
        wandb.log({"eval_mean_reward": mean_reward}, sync=False)
        wandb.log({"eval_std_reward": std_reward}, sync=False)
        wandb.log({"eval_episode_reward": episode_rewards}, sync=False)
        wandb.log({"eval_episode_length": episode_lengths}, sync=False)
        if mean_reward > best_mean_reward:
            print("New best mean reward!")
            # model.save(os.path.join(wandb.run.dir, 'best_model'))
            best_mean_reward = mean_reward
            wandb.log({"best_eval_mean_reward": best_mean_reward}, sync=False)
        if not env.envs[0].move_roll_window():
            break
        idx +=1
