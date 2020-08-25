import torch
import numpy as np
from torch import nn
import gym


class GRUNet(nn.Module):
    def __init__(self, state_space, action_shape, agent_state, net_type):
        super().__init__()
        assert len(action_shape) == 1
        self.agent_state = agent_state
        self.net_type=net_type
        rnn_input_size = np.prod(state_space.spaces['stock_obs'].shape[-2:])
        self.rnn = nn.GRU(input_size=rnn_input_size, hidden_size=rnn_input_size / 4,
                          num_layers=2, batch_first=True)
        if agent_state:
            stock_pos_input_size = np.prod(state_space.spaces['stock_position'].shape)

            self.stock_pos_fc = nn.Sequential(
                nn.Linear(in_features=stock_pos_input_size, out_features=stock_pos_input_size / 2),
                nn.ReLU())
            body_input_size = rnn_input_size / 4 + stock_pos_input_size / 2 + np.prod(state_space.spaces['money'].shape)
        else:
            body_input_size = rnn_input_size / 4
        self.body = nn.Sequential(
            nn.Linear(in_features=body_input_size, out_features=action_shape[0]),
            nn.Tanh(),
        )
        if net_type=='actor':
            self.output1 = nn.Sequential(
                nn.Linear(in_features=action_shape[0], out_features=action_shape[0] - 1),
                nn.Softmax()
            )
        self.output2 = nn.Sequential(
            nn.Linear(in_features=action_shape[0], out_features=1),
            nn.Sigmoid()
        )

    def forward(self, obs, state=None, info={}):
        stock_obs = obs['stock_obs']
        if self.agent_state:
            stock_position = obs['stock_position']
            money = obs['money']
        if not isinstance(obs, torch.Tensor):
            stock_obs = torch.tensor(stock_obs, dtype=torch.float)
            if self.agent_state:
                stock_position = torch.tensor(stock_position, dtype=torch.float)
                money = torch.tensor(money, dtype=torch.float)
        batch, time, stocks, feature = stock_obs.shape
        stock_obs = stock_obs.view(batch, time, -1)
        # batch, feature
        stock_obs = self.rnn(stock_obs)
        if self.agent_state:
            stock_position = stock_position.view(stock_position.shape[0], -1)
            stock_position = self.stock_pos_fc(stock_position)
            hidden = torch.cat((stock_obs, stock_position, money), dim=1)
        else:
            hidden = stocks
        hidden = self.body(hidden)
        logits = self.output2(hidden)
        if self.net_type=='actor':
            stock_ratio = self.output1(hidden)
            logits = torch.cat((stock_ratio, logits), dim=1)
        return logits, state
