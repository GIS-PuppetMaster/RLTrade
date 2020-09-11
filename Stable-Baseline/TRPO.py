import json
from stable_baselines import TRPO
from stable_baselines.common.env_checker import check_env
from Env.TradeEnv import TradeEnv
from Util.Callback import CustomCallback
from Util.BestModelCallback import *
from stable_baselines.common.callbacks import *
import shutil
from Util.CustomPolicy import CustomMultiStockPolicy
import argparse


def make_env(config, seed, mode):
    env_config = config['env'][mode]
    def get_env():
        env = TradeEnv(config=config, **env_config)
        env.seed(seed)
        check_env(env)
        return env

    return get_env


if __name__ == '__main__':
    argparse = argparse.ArgumentParser()
    argparse.add_argument('--config', type=str, default=None)
    args = argparse.parse_args()
    with open(args.config, 'r', encoding='utf-8') as f:
        conf = json.load(f)
    if conf['global_wandb']:
        wandb.init(project='Stable-BaselineTradingV3', sync_tensorboard=True, config=conf)

    os.environ["CUDA_VISIBLE_DEVICES"] = conf['train']['gpu']
    eval_env = TradeEnv(config=conf, **conf['env']['test'])
    eval_env.seed(conf['seed'])
    env = DummyVecEnv([make_env(conf, conf['seed'],'train') for _ in range(conf['env']['train_env_num'])])
    monitorCallback = CustomCallback()
    checkPointCallback = CheckpointCallback(save_freq=conf['train']['save_freq'], save_path=os.path.join(wandb.run.dir,
                                                                                                conf['train'][
                                                                                                    'save_dir']))
    saveBestCallback = MyEvalCallback(eval_env, best_model_save_path=wandb.run.dir, **conf['eval'])
    model = TRPO(CustomMultiStockPolicy, env, verbose=1, tensorboard_log=conf['train']['log_dir'], seed=conf['seed'])
    model.learn(total_timesteps=conf['train']['total_timesteps'], callback=[checkPointCallback, saveBestCallback])
    model.save(os.path.join(wandb.run.dir, 'final_model'))
