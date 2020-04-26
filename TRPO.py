from stable_baselines import *
from stable_baselines.common.env_checker import check_env
from TradeEnv import TradeEnv
from Util.Callback import CustomCallback
from Util.BestModelCallback import *
from stable_baselines.common.callbacks import *
from Util.CustomPolicy import CustomPolicy
import sys
from Config import *




def make_env():
    env = TradeEnv(**train_env_config)
    check_env(env)
    return env


if __name__ == '__main__':
    init_wandb()
    eval_env = TradeEnv(**eval_env_config)
    eval_env.seed(0)
    env = DummyVecEnv([make_env for _ in range(n_training_envs)])
    del_file('E:\运行结果\TRPO\TRPO/train')
    monitorCallback = CustomCallback()
    checkpointPath = './checkpoints/' + net_type
    if os.path.exists(checkpointPath):
        com = input("checkpoint目录：" + checkpointPath + " 已经存在，是否清空[y/n]")
        if com == 'y':
            if del_file(checkpointPath):
                print("清空成功")
            else:
                print("清空失败")
                sys.exit()
    if not os.path.exists(checkpointPath):
        os.makedirs(checkpointPath)
    checkPointCallback = CheckpointCallback(save_freq=10 * EP_LEN, save_path=checkpointPath)
    saveBestCallback = MyEvalCallback(eval_env, best_model_save_path='./BestModels/' + net_type + '/', n_eval_episodes=10,
                                    eval_freq=EP_LEN * 10)
    model = TRPO(CustomPolicy, env, verbose=1, tensorboard_log="./log/", seed=0, policy_kwargs=policy_args)
    # model = TRPO.load(policy=CustomPolicy, env=env, load_path='./model', verbose=1, tensorboard_log="./log/", seed=0)
    wandb.save("model")
    model.learn(total_timesteps=episode * EP_LEN, callback=[monitorCallback, checkPointCallback, saveBestCallback])
    model.save(os.path.join(wandb.run.dir, "model"))
    # model.save("model")
