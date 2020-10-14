import torch
import numpy as np
from torch import nn
from torch.nn.modules import *
from Tianshou.Net.NBeats import NBeatsNet


def del_batch_norm(layers, batch_norm):
    if not batch_norm:
        return filter(lambda x: not (isinstance(x, BatchNorm1d) and isinstance(x, BatchNorm2d) and isinstance(x, BatchNorm3d)), layers)
    else:
        return layers


class StockActor(nn.Module):
    def __init__(self, state_space, action_shape, agent_state, gpu, feature_extract, batch_norm, **kwargs):
        super().__init__()
        assert len(action_shape) == 1
        self.agent_state = agent_state
        if not self.agent_state:
            state_space = {'stock_obs': state_space}
        self.gpu = gpu
        self.feature_extract = feature_extract
        self.batch_norm = batch_norm
        if self.gpu:
            self.device = torch.device('cuda:0')
            kwargs['device'] = self.device
        stock_obs_shape = state_space['stock_obs'].shape
        if feature_extract == 'nbeats':
            self.feature_extract_layer = NBeatsNet(backcast_length=stock_obs_shape[0], **kwargs)
            stock_obs_shape = list(stock_obs_shape)
            f = lambda i: stock_obs_shape[i] if i != 0 else kwargs['forecast_length']
            feature_extract_output_shape = np.prod([f(i) for i in range(len(stock_obs_shape))])
        elif feature_extract == 'gru':
            rnn_input_size = np.prod(state_space['stock_obs'].shape[-2:])
            self.feature_extract_layer = GRU(input_size=rnn_input_size, hidden_size=rnn_input_size // 4,
                                             num_layers=2, batch_first=True)
            feature_extract_output_shape = rnn_input_size // 4
        elif feature_extract == 'conv1d':
            feature_extract_output_shape = 256
            layers = [
                Conv1d(in_channels=stock_obs_shape[-1] * stock_obs_shape[-2], out_channels=32, kernel_size=9, dilation=1),
                BatchNorm1d(32),
                LeakyReLU(),
                Conv1d(in_channels=32, out_channels=32, kernel_size=7, dilation=2),
                BatchNorm1d(32),
                LeakyReLU(),
                Conv1d(in_channels=32, out_channels=64, kernel_size=5, dilation=4),
                BatchNorm1d(64),
                LeakyReLU(),
                Conv1d(in_channels=64, out_channels=128, kernel_size=3, dilation=8),
                BatchNorm1d(128),
                LeakyReLU(),
                Conv1d(in_channels=128, out_channels=feature_extract_output_shape, kernel_size=8, dilation=1),
                BatchNorm1d(256),
                LeakyReLU(),
            ]
            self.feature_extract_layer = Sequential(*del_batch_norm(layers, self.batch_norm))
        else:
            raise Exception(f'Wrong feature_extract type:{feature_extract}')
        if self.agent_state:
            stock_pos_input_size = np.prod(state_space['stock_position'].shape)
            layers = [Linear(in_features=stock_pos_input_size, out_features=stock_pos_input_size // 2),
                      BatchNorm1d(stock_pos_input_size // 2),
                      Tanh()]
            self.stock_pos_fc = Sequential(*del_batch_norm(layers, self.batch_norm))
            body_input_size = feature_extract_output_shape + stock_pos_input_size // 2 + np.prod(
                state_space['money'].shape)
        else:
            body_input_size = feature_extract_output_shape
        layers = [Linear(in_features=body_input_size, out_features=action_shape[0]),
                  BatchNorm1d(action_shape[0]),
                  ReLU()]
        self.body = Sequential(*del_batch_norm(layers, self.batch_norm))
        layers = [Linear(in_features=action_shape[0], out_features=action_shape[0]),
                  BatchNorm1d(action_shape[0]),
                  Tanh(),
                  Linear(in_features=action_shape[0], out_features=action_shape[0] - 1),
                  Tanh()]
        self.output1 = Sequential(*Sequential(*del_batch_norm(layers, self.batch_norm)))
        layers = [Linear(in_features=action_shape[0], out_features=action_shape[0]),
                  BatchNorm1d(action_shape[0]),
                  PReLU(),
                  Linear(in_features=action_shape[0], out_features=1),
                  Sigmoid()]
        self.output2 = Sequential(*del_batch_norm(layers, self.batch_norm))

    def pre_process(self, obs, state=None, info=None):
        if info is None:
            info = {}
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

    def get_hidden(self, obs, state=None, info=None):
        if info is None:
            info = {}
        if self.agent_state:
            stock_obs, stock_position, money = self.pre_process(obs, state, info)
        else:
            if not isinstance(obs, dict):
                obs = {'stock_obs': obs}
            stock_obs = self.pre_process(obs, state, info)
        batch, time, stocks, feature = stock_obs.shape
        # batch, feature
        if self.feature_extract == 'nbeats':
            stock_obs = self.feature_extract_layer(stock_obs)[1].view(batch, -1)
        elif self.feature_extract == 'gru':
            stock_obs = self.feature_extract_layer(stock_obs.view(batch, time, -1))[0][:, -1, :]
        elif self.feature_extract == 'conv1d':
            stock_obs = self.feature_extract_layer(stock_obs.view(batch, stock_obs.shape[1], -1).transpose(1, 2)).transpose(1, 2).view(batch, -1)
        if self.agent_state:
            stock_position = stock_position.view(stock_position.shape[0], -1)
            stock_position = self.stock_pos_fc(stock_position)
            hidden = torch.cat((stock_obs, stock_position, money), dim=1)
        else:
            hidden = stock_obs
        hidden = self.body(hidden)
        return hidden

    def get_logits(self, hidden):
        logits = self.output2(hidden)
        stock_ratio = self.output1(hidden)
        logits = torch.cat((stock_ratio, logits), dim=1)
        return logits

    def forward(self, obs, state=None, info=None):
        if info is None:
            info = {}
        hidden = self.get_hidden(obs, state, info)
        logits = self.get_logits(hidden)
        return logits, state


class StockCritic(Module):
    def __init__(self, state_space, action_shape, agent_state, gpu, feature_extract, with_action, batch_norm, **kwargs):
        super().__init__()
        assert len(action_shape) == 1
        self.agent_state = agent_state
        if not self.agent_state:
            state_space = {'stock_obs': state_space}
        self.with_action = with_action
        self.gpu = gpu
        self.batch_norm = batch_norm
        self.feature_extract = feature_extract
        if self.gpu:
            self.device = torch.device('cuda:0')
            kwargs['device'] = self.device
        stock_obs_shape = state_space['stock_obs'].shape
        if feature_extract == 'nbeats':
            self.feature_extract_layer = NBeatsNet(backcast_length=stock_obs_shape[0], **kwargs)
            stock_obs_shape = list(stock_obs_shape)
            f = lambda i: stock_obs_shape[i] if i != 0 else kwargs['forecast_length']
            feature_extract_output_shape = np.prod([f(i) for i in range(len(stock_obs_shape))])
        elif feature_extract == 'gru':
            rnn_input_size = np.prod(state_space['stock_obs'].shape[-2:])
            self.feature_extract_layer = GRU(input_size=rnn_input_size, hidden_size=rnn_input_size // 4,
                                             num_layers=2, batch_first=True)
            feature_extract_output_shape = rnn_input_size // 4
        elif feature_extract == 'conv1d':
            feature_extract_output_shape = 256
            layers = [
                Conv1d(in_channels=stock_obs_shape[-1] * stock_obs_shape[-2], out_channels=32, kernel_size=9, dilation=1),
                BatchNorm1d(32),
                LeakyReLU(),
                Conv1d(in_channels=32, out_channels=32, kernel_size=7, dilation=2),
                BatchNorm1d(32),
                LeakyReLU(),
                Conv1d(in_channels=32, out_channels=64, kernel_size=5, dilation=4),
                BatchNorm1d(64),
                LeakyReLU(),
                Conv1d(in_channels=64, out_channels=128, kernel_size=3, dilation=8),
                BatchNorm1d(128),
                LeakyReLU(),
                Conv1d(in_channels=128, out_channels=feature_extract_output_shape, kernel_size=8, dilation=1),
                BatchNorm1d(256),
                LeakyReLU(),
            ]
            self.feature_extract_layer = Sequential(*del_batch_norm(layers, self.batch_norm))
        else:
            raise Exception(f'Wrong feature_extract type:{feature_extract}')
        if self.agent_state:
            stock_pos_input_size = np.prod(state_space['stock_position'].shape)
            layers = [
                Linear(in_features=stock_pos_input_size, out_features=stock_pos_input_size // 2),
                BatchNorm1d(stock_pos_input_size // 2),
                Tanh()]
            self.stock_pos_fc = Sequential(*del_batch_norm(layers, self.batch_norm))
            body_input_size = feature_extract_output_shape + stock_pos_input_size // 2 + np.prod(
                state_space['money'].shape)
        else:
            body_input_size = feature_extract_output_shape
        if self.with_action:
            body_input_size += action_shape[0]
        layers = [Linear(in_features=body_input_size, out_features=256),
                  BatchNorm1d(256),
                  Tanh(),
                  Linear(in_features=256, out_features=256),
                  BatchNorm1d(256),
                  Tanh(),
                  Linear(in_features=256, out_features=1),
                  Tanh()
                  ]
        self.output = Sequential(*del_batch_norm(layers, self.batch_norm))

    def pre_process(self, obs, action=None, state=None, info=None):
        if info is None:
            info = {}
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

    def get_hidden(self, obs, action=None, state=None, info=None):
        if info is None:
            info = {}
        if self.agent_state:
            stock_obs, stock_position, money, action = self.pre_process(obs, action, state, info)
        else:
            if not isinstance(obs, dict):
                obs = {'stock_obs': obs}
            stock_obs, action = self.pre_process(obs, action, state, info)
        batch, time, stocks, feature = stock_obs.shape
        # batch, feature
        if self.feature_extract == 'nbeats':
            stock_obs = self.feature_extract_layer(stock_obs)[1].view(batch, -1)
        elif self.feature_extract == 'gru':
            stock_obs = self.feature_extract_layer(stock_obs.view(batch, time, -1))[0][:, -1, :]
        elif self.feature_extract == 'conv1d':
            stock_obs = self.feature_extract_layer(stock_obs.view(batch, stock_obs.shape[1], -1).transpose(1, 2)).transpose(1, 2).view(batch, -1)
        if self.agent_state:
            stock_position = stock_position.view(stock_position.shape[0], -1)
            stock_position = self.stock_pos_fc(stock_position)
            hidden = torch.cat((stock_obs, stock_position, money), dim=1)
        else:
            hidden = stock_obs
        return hidden, action

    def get_logits(self, hidden, action=None):
        if self.with_action:
            hidden = torch.cat((hidden, action), dim=1)
        logits = self.output(hidden)
        return logits

    def forward(self, obs, action=None, state=None, info=None):
        if info is None:
            info = {}
        hidden, action = self.get_hidden(obs, action, state, info)
        logits = self.get_logits(hidden, action)
        return logits


class StockDistributionalActor(StockActor):
    def __init__(self, state_space, action_shape, agent_state, gpu, feature_extract, batch_norm, **kwargs):
        super().__init__(state_space, action_shape, agent_state, gpu, feature_extract, batch_norm, **kwargs)
        layers = [Linear(in_features=action_shape[0], out_features=action_shape[0]),
                  BatchNorm1d(action_shape[0]),
                  ReLU(),
                  Linear(in_features=action_shape[0], out_features=action_shape[0]),
                  Sigmoid()]
        self.output_std = Sequential(*del_batch_norm(layers, self.batch_norm))

    def forward(self, obs, state=None, info=None):
        if info is None:
            info = {}
        hidden = self.get_hidden(obs, state, info)
        logits = self.get_logits(hidden)
        logits_std = self.output_std(hidden)
        return (logits, logits_std), state
