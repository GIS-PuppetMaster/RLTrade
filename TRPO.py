from stable_baselines import *
from stable_baselines.common.env_checker import check_env
from TradeEnv import TradeEnv
from Util.Callback import CustomCallback
from Util.BestModelCallback import *
from stable_baselines.common.callbacks import *
from Util.Util import find_model
import wandb
import shutil


def make_env():
    env = TradeEnv(**train_env_config)
    env.seed(seed)
    check_env(env)
    return env


if __name__ == '__main__':
    load_checkpoint = False
    load_id = ""
    timestamp = ""
    model_path = None
    if load_id != "":
        load_checkpoint = True
    if load_checkpoint:
        folder_name, model_path, _ = find_model(id=load_id, useVersion="last", timestamp=timestamp)
        try:
            # 备份原来根目录下的Config.py
            os.rename('./Config.py', './Config_old.py')
        except:
            pass
        # 使用id下的Config.py
        shutil.copyfile(os.path.join('./wandb', folder_name, 'Config.py'), './Config.py',)
        # from Config import GPU, eval_env_config, config, train_env_config, n_training_envs, save_freq, n_eval_episodes, \
        #     eval_freq, policy_args, episode, EP_LEN, seed
        from Config import *
        wandb.init(project='Stable-BaselineTradingV2', sync_tensorboard=True, config=config, id=load_id, resume="must")
    else:
        from Config import *

        wandb.init(project='Stable-BaselineTradingV2', sync_tensorboard=True, config=config)
        shutil.copyfile('./Config.py', os.path.join(wandb.run.dir, 'Config.py'))
    from Util.CustomPolicy import CustomPolicy

    os.environ["CUDA_VISIBLE_DEVICES"] = GPU
    eval_env = TradeEnv(**eval_env_config)
    eval_env.seed(seed)
    env = DummyVecEnv([make_env for _ in range(n_training_envs)])
    del_file('E:\运行结果\TRPO\TRPO/train')
    monitorCallback = CustomCallback()
    checkPointCallback = CheckpointCallback(save_freq=save_freq, save_path=os.path.join(wandb.run.dir, 'checkpoints'))
    saveBestCallback = MyEvalCallback(eval_env, best_model_save_path=wandb.run.dir,
                                      n_eval_episodes=n_eval_episodes,
                                      eval_freq=eval_freq)
    model = TRPO(CustomPolicy, env, verbose=1, tensorboard_log="./log/", seed=seed, policy_kwargs=policy_args)
    if load_checkpoint:
        print("载入断点:{}".format(model_path))
        model.load(model_path)
    model.learn(total_timesteps=episode * EP_LEN, callback=[checkPointCallback, saveBestCallback],
                reset_num_timesteps=not load_checkpoint)
    model.save(os.path.join(wandb.run.dir, 'final_model'))
