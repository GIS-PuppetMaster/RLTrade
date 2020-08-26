import tianshou as ts
from Env.TradeEnv import TradeEnv
from Tianshou.Net.MultiStockTradeNet import *
from torch.utils.tensorboard import SummaryWriter
import json
import argparse

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--config', type=str)
    args = argparser.parse_args()

    with open(args.config, 'r', encoding='utf-8') as f:
        config = json.load(f)

    save_dir = config['train']['save_dir']
    if config['global_wandb']:
        import wandb

        wandb.init(**config['wandb'], config=config)
        save_dir = wandb.run.dir + "\\"
    train_envs = ts.env.SubprocVectorEnv(
        [lambda: TradeEnv(**config['env']['train']) for _ in range(config['env']['train_env_num'])])
    test_env = ts.env.SubprocVectorEnv(
        [lambda: TradeEnv(**config['env']['test']) for _ in range(config['env']['test_env_num'])])

    state_space = train_envs.observation_space[0]
    action_shape = train_envs.action_space[0].shape

    actor_net = GRUActor(state_space, action_shape, config['env']['train']['agent_state'], config['train']['gpu'])
    actor_optim = torch.optim.Adam(actor_net.parameters(), lr=config['policy']['actor_lr'])
    critic1_net = GRUCritic(state_space, action_shape, config['env']['train']['agent_state'], config['train']['gpu'])
    critic1_optim = torch.optim.Adam(critic1_net.parameters(), lr=config['policy']['critic_1_lr'])
    critic2_net = GRUCritic(state_space, action_shape, config['env']['train']['agent_state'], config['train']['gpu'])
    critic2_optim = torch.optim.Adam(critic2_net.parameters(), lr=config['policy']['critic_2_lr'])

    if config['train']['gpu']:
        actor_net = actor_net.cuda()
        critic1_net = critic1_net.cuda()
        critic2_net = critic2_net.cuda()

    policy = ts.policy.TD3Policy(actor_net, actor_optim, critic1_net, critic1_optim, critic2_net, critic2_optim,
                                 **config['policy']['policy_parameter'],
                                 action_range=(train_envs.action_space[0].low.mean(), train_envs.action_space[0].high.mean()))

    train_collector = ts.data.Collector(policy, train_envs,
                                        ts.data.PrioritizedReplayBuffer(**config['train']['replay_buffer']))
    test_collector = ts.data.Collector(policy, test_env)

    writer = SummaryWriter(config['train']['log_dir'])
    result = ts.trainer.offpolicy_trainer(policy, train_collector, test_collector, **config['train']['train_parameter'],
                                          writer=writer)
    torch.save(policy.state_dict(), save_dir)
