import json
import sys
import wandb

if sys.platform == 'win32':
    sys.path.append("D:\\PycharmProjects\\Stable-BaselineTrading\\")
else:
    sys.path.append("/usr/zkx/Stable-BaselineTrading/")
from stable_baselines import TRPO
from stable_baselines.common.env_checker import check_env
from Env.TradeEnv import TradeEnv
from Util.Callback import CustomCallback
from Util.BestModelCallback import *
from stable_baselines.common.callbacks import *
from stable_baselines.common.vec_env import *
import shutil
from Util.CustomPolicy import CustomMultiStockPolicy
import argparse


def make_env(config, seed, mode, id):
    env_config = config['env'][mode]

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
    argparse = argparse.ArgumentParser()
    argparse.add_argument('--config', type=str, default=None)
    args = argparse.parse_args()
    with open(args.config, 'r', encoding='utf-8') as f:
        conf = json.load(f)
    conf['env']['train']['result_path'] = replace_path(conf['env']['train']['result_path'], name)
    conf['env']['test']['result_path'] = replace_path(conf['env']['test']['result_path'], name)
    if conf['global_wandb']:
        wandb.init(project='Stable-BaselineTradingV3', sync_tensorboard=True, config=conf, tensorboard=True)

    os.environ["CUDA_VISIBLE_DEVICES"] = conf['train']['gpu']
    eval_env = TradeEnv(config=conf, **conf['env']['test'], env_id=0)
    if conf['seed'] is not None:
        eval_env.seed(conf['seed'])
    env = DummyVecEnv([make_env(conf, conf['seed'], 'train', id) for id in range(conf['env']['train_env_num'])])
    monitorCallback = CustomCallback()
    checkPointCallback = CheckpointCallback(save_freq=conf['train']['save_freq'], save_path=os.path.join(wandb.run.dir, conf['train']['save_dir']))
    saveBestCallback = MyEvalCallback(eval_env, best_model_save_path=wandb.run.dir, **conf['eval'])
    model = TRPO(CustomMultiStockPolicy, env, tensorboard_log=conf['train']['log_dir'], seed=conf['seed'], **conf['policy'])
    model.learn(total_timesteps=conf['train']['total_timesteps'], callback=[checkPointCallback, saveBestCallback])
    model.save(os.path.join(wandb.run.dir, 'final_model'))
