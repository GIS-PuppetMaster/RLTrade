import torch
import numpy as np
from torch import nn
import importlib

from Tianshou.Net.NBeats import NBeatsNet


class GRUActor(nn.Module):
    def __init__(self, state_space, action_shape, agent_state, gpu, forecast_length, stack_types, nb_blocks_pre_stack,
                 thetas_dims, shard_weights_in_stack, hidden_layer_units, **kwargs):
        super().__init__()
        assert len(action_shape) == 1
        self.agent_state = agent_state
        self.gpu = gpu
        if self.gpu:
            self.device = torch.device('cuda:0')
        stock_obs_shape = state_space['stock_obs'].shape
        self.forecast = NBeatsNet(backcast_length=stock_obs_shape[0], forecast_length=forecast_length,
                                  stack_types=stack_types,
                                  nb_blocks_per_stack=nb_blocks_pre_stack,
                                  thetas_dims=thetas_dims, share_weights_in_stack=shard_weights_in_stack,
                                  hidden_layer_units=hidden_layer_units, device=self.device)
        stock_obs_shape = list(stock_obs_shape)
        f = lambda i: stock_obs_shape[i] if i != 0 else forecast_length
        forecast_output_shape = np.prod([f(i) for i in range(len(stock_obs_shape))])
        if agent_state:
            stock_pos_input_size = np.prod(state_space['stock_position'].shape)
            self.stock_pos_fc = nn.Sequential(
                nn.Linear(in_features=stock_pos_input_size, out_features=stock_pos_input_size // 2),
                nn.ReLU())
            body_input_size = forecast_output_shape + stock_pos_input_size // 2 + np.prod(state_space['money'].shape)
        else:
            body_input_size = forecast_output_shape
        self.body = nn.Sequential(
            nn.Linear(in_features=body_input_size, out_features=action_shape[0]),
            nn.Tanh(),
        )
        self.output1 = nn.Sequential(
            nn.Linear(in_features=action_shape[0], out_features=action_shape[0] - 1),
            nn.Sigmoid()
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
        if self.gpu:
            stock_obs = stock_obs.cuda()
            if self.agent_state:
                stock_position = stock_position.cuda()
                money = money.cuda()
        batch, time, stocks, feature = stock_obs.shape
        # batch, feature
        stock_obs = self.forecast(stock_obs)[1].view(batch, -1)
        if self.agent_state:
            stock_position = stock_position.view(stock_position.shape[0], -1)
            stock_position = self.stock_pos_fc(stock_position)
            hidden = torch.cat((stock_obs, stock_position, money), dim=1)
        else:
            hidden = stocks
        hidden = self.body(hidden)
        logits = self.output2(hidden)
        stock_ratio = self.output1(hidden)
        logits = torch.cat((stock_ratio, logits), dim=1)
        return torch.Tensor.cpu(logits), state


class GRUCritic(nn.Module):
    def __init__(self, state_space, action_shape, agent_state, gpu, forecast_length, stack_types, nb_blocks_pre_stack,
                 thetas_dims, shard_weights_in_stack, hidden_layer_units, **kwargs):
        super().__init__()
        assert len(action_shape) == 1
        self.agent_state = agent_state
        self.gpu = gpu
        if self.gpu:
            self.device = torch.device('cuda:0')
        stock_obs_shape = state_space['stock_obs'].shape
        self.forecast = NBeatsNet(backcast_length=stock_obs_shape[0], forecast_length=forecast_length,
                                  stack_types=stack_types,
                                  nb_blocks_per_stack=nb_blocks_pre_stack,
                                  thetas_dims=thetas_dims, share_weights_in_stack=shard_weights_in_stack,
                                  hidden_layer_units=hidden_layer_units, device=self.device)
        stock_obs_shape = list(stock_obs_shape)
        f = lambda i: stock_obs_shape[i] if i != 0 else forecast_length
        forecast_output_shape = np.prod([f(i) for i in range(len(stock_obs_shape))])
        if agent_state:
            stock_pos_input_size = np.prod(state_space['stock_position'].shape)

            self.stock_pos_fc = nn.Sequential(
                nn.Linear(in_features=stock_pos_input_size, out_features=stock_pos_input_size // 2),
                nn.ReLU())
            body_input_size = forecast_output_shape + stock_pos_input_size // 2 + np.prod(state_space['money'].shape)
        else:
            body_input_size = forecast_output_shape
        body_input_size += action_shape[0]
        self.output = nn.Sequential(
            nn.Linear(in_features=body_input_size, out_features=256),
            nn.Tanh(),
            nn.Linear(in_features=256, out_features=256),
            nn.Tanh(),
            nn.Linear(in_features=256, out_features=1),
        )

    def forward(self, obs, action, state=None, info={}):
        stock_obs = obs['stock_obs']
        if self.agent_state:
            stock_position = obs['stock_position']
            money = obs['money']
        if not isinstance(obs, torch.Tensor):
            stock_obs = torch.tensor(stock_obs, dtype=torch.float)
            if self.agent_state:
                stock_position = torch.tensor(stock_position, dtype=torch.float)
                money = torch.tensor(money, dtype=torch.float)
        if not isinstance(action, torch.Tensor):
            action = torch.tensor(action, dtype=torch.float)
        if self.gpu:
            stock_obs = stock_obs.cuda()
            action = action.cuda()
            if self.agent_state:
                stock_position = stock_position.cuda()
                money = money.cuda()
        batch, time, stocks, feature = stock_obs.shape
        # batch, feature
        stock_obs = self.forecast(stock_obs)[1].view(batch, -1)
        if self.agent_state:
            stock_position = stock_position.view(stock_position.shape[0], -1)
            stock_position = self.stock_pos_fc(stock_position)
            hidden = torch.cat((stock_obs, stock_position, money), dim=1)
        else:
            hidden = stocks
        hidden = torch.cat((hidden, action), dim=1)
        logits = self.output(hidden)
        return torch.Tensor.cpu(logits)
