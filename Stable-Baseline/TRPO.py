from stable_baselines import TRPO
from stable_baselines.common.env_checker import check_env
from Env.TradeEnv import TradeEnv
from Util.Callback import CustomCallback
from Util.BestModelCallback import *
from stable_baselines.common.callbacks import *
import shutil
from Config import *
from Util.CustomPolicy import CustomPolicy
import argparse

def make_env():
    env = TradeEnv(**train_env_config)
    env.seed(seed)
    check_env(env)
    return env


if __name__ == '__main__':
    argparse = argparse.ArgumentParser()
    argparse.add_argument('--load_checkpoint', type=bool, default=False)
    argparse.add_argument('--with_wandb', type=bool, default=True)
    argparse.add_argument('--load_id', type=str, default='')
    argparse.add_argument('--timestamp', type=str, default='')
    argparse.add_argument('--model_path', type=str, default=None)
    args = argparse.parse_args()
    load_checkpoint = args.load_checkpoint
    load_id = args.load_id
    timestamp = args.timestamp
    model_path = args.model_path
    if load_id != "" and not load_checkpoint:
        raise Exception(f"load_id:{load_id} doesn't match load_checkpoint:{load_checkpoint}")
    if load_checkpoint:
        folder_name, model_path, _ = find_model(id=load_id, useVersion="last", timestamp=timestamp)
        import yaml

        with open(os.path.join('../wandb', folder_name, 'config.yaml'), 'r') as f:
            conf = f.read()
        conf = yaml.load(conf)
        conf['agent_config'] = conf['agent_config']['value']
        conf['train_env_config'] = conf['train_env_config']['value']
        conf['eval_env_config'] = conf['eval_env_config']['value']
        if conf['train_env_config']['post_processor'] == 'post_processor':
            conf['train_env_config']['post_processor'] = post_processor
        else:
            raise Exception(
                "train_env_config:post_processor为不支持的类型{}".format(conf['train_env_config']['post_processor']))
        if conf['eval_env_config']['post_processor'] == 'post_processor':
            conf['eval_env_config']['post_processor'] = post_processor
        else:
            raise Exception("eval_env_config:post_processor为不支持的类型{}".format(conf['eval_env_config']['post_processor']))
        globals().update(conf)
        globals().update(conf['agent_config'])
        if args.with_wandb:
            wandb.init(project='Stable-BaselineTradingV2', sync_tensorboard=True, config=config, id=load_id, resume="must")
    else:
        if args.with_wandb:
            wandb.init(project='Stable-BaselineTradingV2', sync_tensorboard=True, config=config)
        shutil.copyfile('../Config.py', os.path.join(wandb.run.dir, 'Config.py'))

    os.environ["CUDA_VISIBLE_DEVICES"] = GPU
    eval_env = TradeEnv(**eval_env_config)
    eval_env.seed(seed)
    env = DummyVecEnv([make_env for _ in range(n_training_envs)])
    del_file('E:\运行结果\TRPO\TRPO/train')
    monitorCallback = CustomCallback()
    checkPointCallback = CheckpointCallback(save_freq=save_freq, save_path=os.path.join(wandb.run.dir,
                                                                                        '../checkpoints'))
    saveBestCallback = MyEvalCallback(eval_env, best_model_save_path=wandb.run.dir,
                                      n_eval_episodes=n_eval_episodes,
                                      eval_freq=eval_freq)
    model = TRPO(CustomPolicy, env, verbose=1, tensorboard_log="./log/", seed=seed, policy_kwargs=policy_args)
    # model = DDPG(policy=MlpPolicy ,env=env, verbose=1, tensorboard_log="./log/", seed=seed)
    if load_checkpoint:
        print("载入断点:{}".format(model_path))
        model.load(model_path, env=env)
    model.learn(total_timesteps=episode * EP_LEN, callback=[checkPointCallback, saveBestCallback],
                reset_num_timesteps=not load_checkpoint)
    model.save(os.path.join(wandb.run.dir, 'final_model'))
