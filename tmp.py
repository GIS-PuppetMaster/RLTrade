import json
from stable_baselines.common.vec_env import *
from stable_baselines.common.env_checker import check_env
from Env.TradeEnv import TradeEnv
import os


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
    return os.path.join(tup[0], insert_fold, tup[1], "",)

if __name__ == '__main__':
    name = input("please input run_name:\n")
    with open('./Config/TRPOConfigLinux.json', 'r', encoding='utf-8') as f:
        conf = json.load(f)

    a = replace_path(conf['env']['train']['result_path'], name)
    conf['env']['test']['result_path'] += f'{name}/'
    env = DummyVecEnv([make_env(conf, conf['seed'], 'train', id) for id in range(conf['env']['train_env_num'])])
