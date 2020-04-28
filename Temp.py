import os
import wandb
from stable_baselines import *
from stable_baselines.common.callbacks import *
from TradeEnv import *
from Config import *

path = './checkpoints/small_net_5stocks_regularize_StandardScaler/'
model_list = os.listdir(path)
step_list = []
for file in model_list:
    step_list.append(int(file.split('_')[-2]))
step_list = np.array(step_list)
step_list.sort()
wandb.init(project='Stable-BaselineTradingV2', sync_tensorboard=True, config=config, resume='must', id='7u3n40ye')
eval_env = TradeEnv(**eval_env_config)
best_mean_reward = -np.inf
count=0
for step in step_list:
    if step<=7867392:
        continue
    count += 1
    if count%10!=0:
        continue
    file = 'rl_model_'+str(step)+'_steps.zip'
    print(file)
    model = TRPO.load(path + file)
    episode_rewards, episode_lengths = evaluate_policy(model, eval_env,
                                                       n_eval_episodes=20,
                                                       render=False,
                                                       deterministic=True,
                                                       return_episode_rewards=True)
    mean_reward, std_reward = np.mean(episode_rewards), np.std(episode_rewards)
    mean_ep_length, std_ep_length = np.mean(episode_lengths), np.std(episode_lengths)
    wandb.log({"eval_mean_reward": mean_reward}, sync=False, step=step)
    wandb.log({"eval_std_reward": std_reward}, sync=False, step=step)
    wandb.log({"eval_episode_reward": episode_rewards}, sync=False, step=step)
    wandb.log({"eval_episode_length": episode_lengths}, sync=False, step=step)
    if mean_reward > best_mean_reward:
        best_mean_reward = mean_reward
        wandb.log({"best_eval_mean_reward": best_mean_reward}, sync=False, step=step)
