import torch
import numpy as np
from torch import nn
import importlib

from Tianshou.Net.NBeats import NBeatsNet


class StockActor(nn.Module):
    def __init__(self, state_space, action_shape, agent_state, gpu, feature_extract, **kwargs):
        super().__init__()
        assert len(action_shape) == 1
        self.agent_state = agent_state
        if not self.agent_state:
            state_space = {'stock_obs': state_space}
        self.gpu = gpu
        self.feature_extract = feature_extract
        if self.gpu:
            self.device = torch.device('cuda:0')
            kwargs['device'] = self.device
        if feature_extract == 'nbeats':
            stock_obs_shape = state_space['stock_obs'].shape
            self.feature_extract_layer = NBeatsNet(backcast_length=stock_obs_shape[0], **kwargs)
            stock_obs_shape = list(stock_obs_shape)
            f = lambda i: stock_obs_shape[i] if i != 0 else kwargs['forecast_length']
            feature_extract_output_shape = np.prod([f(i) for i in range(len(stock_obs_shape))])
        elif feature_extract == 'gru':
            rnn_input_size = np.prod(state_space['stock_obs'].shape[-2:])
            self.feature_extract_layer = nn.GRU(input_size=rnn_input_size, hidden_size=rnn_input_size // 4,
                                                num_layers=2, batch_first=True)
            feature_extract_output_shape = rnn_input_size // 4
        else:
            raise Exception(f'Wrong feature_extract type:{feature_extract}')
        if self.agent_state:
            stock_pos_input_size = np.prod(state_space['stock_position'].shape)
            self.stock_pos_fc = nn.Sequential(
                nn.Linear(in_features=stock_pos_input_size, out_features=stock_pos_input_size // 2),
                nn.BatchNorm1d(stock_pos_input_size // 2),
                nn.Tanh())
            body_input_size = feature_extract_output_shape + stock_pos_input_size // 2 + np.prod(
                state_space['money'].shape)
        else:
            body_input_size = feature_extract_output_shape
        self.body = nn.Sequential(
            nn.Linear(in_features=body_input_size, out_features=action_shape[0]),
            nn.BatchNorm1d(action_shape[0]),
            nn.ReLU(),
        )
        self.output1 = nn.Sequential(
            nn.Linear(in_features=action_shape[0], out_features=action_shape[0]),
            nn.BatchNorm1d(action_shape[0]),
            nn.Tanh(),
            nn.Linear(in_features=action_shape[0], out_features=action_shape[0] - 1),
            nn.Tanh()
        )
        self.output2 = nn.Sequential(
            nn.Linear(in_features=action_shape[0], out_features=action_shape[0]),
            nn.BatchNorm1d(action_shape[0]),
            nn.PReLU(),
            nn.Linear(in_features=action_shape[0], out_features=1),
            nn.Sigmoid()
        )

    def pre_process(self, obs, state=None, info={}):
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
        if self.agent_state:
            return stock_obs, stock_position, money
        else:
            return stock_obs

    def get_hidden(self, obs, state=None, info={}):
        if self.agent_state:
            stock_obs, stock_position, money = self.pre_process(obs, state, info)
        else:
            obs = {'stock_obs':obs}
            stock_obs = self.pre_process(obs, state, info)
        batch, time, stocks, feature = stock_obs.shape
        # batch, feature
        if self.feature_extract == 'nbeats':
            stock_obs = self.feature_extract_layer(stock_obs)[1].view(batch, -1)
        elif self.feature_extract == 'gru':
            stock_obs = self.feature_extract_layer(stock_obs.view(batch, time, -1))[0][:, -1, :]
        if self.agent_state:
            stock_position = stock_position.view(stock_position.shape[0], -1)
            stock_position = self.stock_pos_fc(stock_position)
            hidden = torch.cat((stock_obs, stock_position, money), dim=1)
        else:
            hidden = stocks
        hidden = self.body(hidden)
        return hidden

    def get_logits(self, hidden):
        logits = self.output2(hidden)
        stock_ratio = self.output1(hidden)
        logits = torch.cat((stock_ratio, logits), dim=1)
        return logits

    def forward(self, obs, state=None, info={}):
        hidden = self.get_hidden(obs, state, info)
        logits = self.get_logits(hidden)
        return logits, state


class StockCritic(nn.Module):
    def __init__(self, state_space, action_shape, agent_state, gpu, feature_extract, with_action, **kwargs):
        super().__init__()
        assert len(action_shape) == 1
        self.agent_state = agent_state
        if not self.agent_state:
            state_space = {'stock_obs': state_space}
        self.with_action = with_action
        self.gpu = gpu
        self.feature_extract = feature_extract
        if self.gpu:
            self.device = torch.device('cuda:0')
            kwargs['device'] = self.device
        if feature_extract == 'nbeats':
            stock_obs_shape = state_space['stock_obs'].shape
            self.feature_extract_layer = NBeatsNet(backcast_length=stock_obs_shape[0], **kwargs)
            stock_obs_shape = list(stock_obs_shape)
            f = lambda i: stock_obs_shape[i] if i != 0 else kwargs['forecast_length']
            feature_extract_output_shape = np.prod([f(i) for i in range(len(stock_obs_shape))])
        elif feature_extract == 'gru':
            rnn_input_size = np.prod(state_space['stock_obs'].shape[-2:])
            self.feature_extract_layer = nn.GRU(input_size=rnn_input_size, hidden_size=rnn_input_size // 4,
                                                num_layers=2, batch_first=True)
            feature_extract_output_shape = rnn_input_size // 4
        else:
            raise Exception(f'Wrong feature_extract type:{feature_extract}')
        if self.agent_state:
            stock_pos_input_size = np.prod(state_space['stock_position'].shape)
            self.stock_pos_fc = nn.Sequential(
                nn.Linear(in_features=stock_pos_input_size, out_features=stock_pos_input_size // 2),
                nn.BatchNorm1d(stock_pos_input_size // 2),
                nn.Tanh())
            body_input_size = feature_extract_output_shape + stock_pos_input_size // 2 + np.prod(
                state_space['money'].shape)
        else:
            body_input_size = feature_extract_output_shape
        if self.with_action:
            body_input_size += action_shape[0]
        self.output = nn.Sequential(
            nn.Linear(in_features=body_input_size, out_features=256),
            nn.BatchNorm1d(256),
            nn.Tanh(),
            nn.Linear(in_features=256, out_features=256),
            nn.BatchNorm1d(256),
            nn.Tanh(),
            nn.Linear(in_features=256, out_features=1),
            nn.Tanh()
        )

    def pre_process(self, obs, action=None, state=None, info={}):
        stock_obs = obs['stock_obs']
        if self.agent_state:
            stock_position = obs['stock_position']
            money = obs['money']
        if not isinstance(obs, torch.Tensor):
            stock_obs = torch.tensor(stock_obs, dtype=torch.float)
            if self.agent_state:
                stock_position = torch.tensor(stock_position, dtype=torch.float)
                money = torch.tensor(money, dtype=torch.float)
        if self.with_action:
            if not isinstance(action, torch.Tensor):
                action = torch.tensor(action, dtype=torch.float)
        if self.gpu:
            stock_obs = stock_obs.cuda()
            if self.with_action:
                action = action.cuda()
            if self.agent_state:
                stock_position = stock_position.cuda()
                money = money.cuda()
        if self.agent_state:
            return stock_obs, stock_position, money, action
        else:
            return stock_obs, action

    def get_hidden(self, obs, action=None, state=None, info={}):
        if self.agent_state:
            stock_obs, stock_position, money, action = self.pre_process(obs, action, state, info)
        else:
            obs = {'stock_obs':obs}
            stock_obs, action = self.pre_process(obs, action, state, info)
        batch, time, stocks, feature = stock_obs.shape
        # batch, feature
        if self.feature_extract == 'nbeats':
            stock_obs = self.feature_extract_layer(stock_obs)[1].view(batch, -1)
        elif self.feature_extract == 'gru':
            stock_obs = self.feature_extract_layer(stock_obs.view(batch, time, -1))[0][:, -1, :]
        if self.agent_state:
            stock_position = stock_position.view(stock_position.shape[0], -1)
            stock_position = self.stock_pos_fc(stock_position)
            hidden = torch.cat((stock_obs, stock_position, money), dim=1)
        else:
            hidden = stocks
        return hidden, action

    def get_logits(self, hidden, action=None):
        if self.with_action:
            hidden = torch.cat((hidden, action), dim=1)
        logits = self.output(hidden)
        return logits

    def forward(self, obs, action=None, state=None, info={}):
        hidden, action = self.get_hidden(obs, action, state, info)
        logits = self.get_logits(hidden, action)
        return logits


class StockDistributionalActor(StockActor):
    def __init__(self, state_space, action_shape, agent_state, gpu, feature_extract, **kwargs):
        super().__init__(state_space, action_shape, agent_state, gpu, feature_extract, **kwargs)
        self.output_std = nn.Sequential(
            nn.Linear(in_features=action_shape[0], out_features=action_shape[0]),
            nn.BatchNorm1d(action_shape[0]),
            nn.ReLU(),
            nn.Linear(in_features=action_shape[0], out_features=action_shape[0]),
            nn.Sigmoid()
        )

    def forward(self, obs, state=None, info={}):
        hidden = self.get_hidden(obs, state, info)
        logits = self.get_logits(hidden)
        logits_std = self.output_std(hidden)
        return (logits, logits_std), state
