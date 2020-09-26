import tianshou
import torch
from typing import Union, Optional, Tuple, Any
import numpy as np
from tianshou.data import Batch, SegmentTree, to_numpy
from tianshou.data import PrioritizedReplayBuffer
from collections import OrderedDict
import pandas as pd
import os
import dill
from Util.Util import *


class StockReplayBuffer(tianshou.data.ReplayBuffer):
    def __init__(self, size: int, **kwargs):
        super().__init__(size)
        assert 'stock_codes' in kwargs.keys()
        assert 'stock_data_path' in kwargs.keys()
        assert 'data_type' in kwargs.keys()
        assert 'post_processor' in kwargs.keys()
        kwargs['post_processor'] = get_modules(kwargs['post_processor'])
        _, stock_data, stock_data_for_state, _ = read_stock_data(**kwargs)

        assert isinstance(stock_data, OrderedDict)
        self.date_list = list(stock_data.keys())
        self.value_list = list(stock_data.values())
        self.state_value_list = list(stock_data_for_state.values())
        self.post_processor = kwargs['post_processor']

    def add(self,
            obs: Union[dict, Batch, np.ndarray, float],
            act: Union[dict, Batch, np.ndarray, float],
            rew: Union[int, float],
            done: Union[bool, int],
            obs_next: Optional[Union[dict, Batch, np.ndarray, float]] = None,
            info: Optional[Union[dict, Batch]] = {},
            policy: Optional[Union[dict, Batch]] = {},
            **kwargs) -> None:
        obs, obs_next = self._convert_obs(obs, obs_next, info)
        super(StockReplayBuffer, self).add(obs, act, rew, done, obs_next, info, policy, **kwargs)

    def _get_stock_obs(self, stock_obs: dict, info: dict) -> np.ndarray:
        obs_pass_date = stock_obs[0]
        obs_current_date = stock_obs[1]
        stock_index = info['stock_index'][0]
        pass_index = self.date_list.index(obs_pass_date)
        current_index = self.date_list.index(obs_current_date)
        data = self.state_value_list[pass_index - 1:current_index - 1]
        # stack time step
        for i in range(len(data)):
            data[i] = np.expand_dims(data[i], axis=0)
        obs = np.concatenate(data, axis=0)
        obs = np.take(obs, stock_index, axis=1)
        if self.post_processor[0].__name__ in force_apply_in_step:
            obs = self.post_processor[0](stock_obs)
        return obs

    def _convert_obs(self, obs: dict, obs_next: Optional[dict], info: dict) -> (np.ndarray, np.ndarray):
        obs_pass_date = info['obs_pass_date']
        obs_current_date = info['obs_current_date']
        obs_next_pass_date = info['obs_next_pass_date']
        obs_next_current_date = info['obs_next_current_date']
        obs['stock_obs'] = np.array([obs_pass_date, obs_current_date])
        obs_next['stock_obs'] = np.array([obs_next_pass_date, obs_next_current_date])

        return obs, obs_next

    def _fetch_obs(self, index: np.ndarray):
        obs = self.get(index, 'obs')
        obs_next = self.get(index, 'obs_next')
        info = self.get(index, 'info')
        obs_replaced = []
        obs_next_replaced = []
        # for each batch
        for i in range(index.shape[0]):
            obs_replaced.append(np.expand_dims(self._get_stock_obs(obs['stock_obs'][i, ...], info), axis=0))
            obs_next_replaced.append(
                np.expand_dims(self._get_stock_obs(obs_next['stock_obs'][i, ...], info), axis=0))
        obs_replaced = np.concatenate(obs_replaced, axis=0)
        obs_next_replaced = np.concatenate(obs_next_replaced, axis=0)
        obs['stock_obs'] = obs_replaced
        obs_next['stock_obs'] = obs_next_replaced
        return obs, obs_next

    def __getitem__(self, index: np.ndarray) -> Batch:
        obs, obs_next = self._fetch_obs(index)
        return Batch(
            obs=obs,
            act=self.act[index],
            rew=self.rew[index],
            done=self.done[index],
            obs_next=obs_next,
            info=self.get(index, 'info'),
            policy=self.get(index, 'policy'),
        )


class StockPrioritizedReplayBuffer(PrioritizedReplayBuffer, StockReplayBuffer):
    """Implementation of Prioritized Experience Replay. arXiv:1511.05952

    :param float alpha: the prioritization exponent.
    :param float beta: the importance sample soft coefficient.

    .. seealso::

        Please refer to :class:`~tianshou.data.ReplayBuffer` for more detailed
        explanation.
    """

    def __init__(self, size: int, alpha: float, beta: float, **kwargs) -> None:
        StockReplayBuffer(size, **kwargs).__init__(size, **kwargs)
        super().__init__(size, alpha, beta, **kwargs)

    def add(self,
            obs: Union[dict, Batch, np.ndarray, float],
            act: Union[dict, Batch, np.ndarray, float],
            rew: Union[int, float],
            done: Union[bool, int],
            obs_next: Optional[Union[dict, Batch, np.ndarray, float]] = None,
            info: Optional[Union[dict, Batch]] = {},
            policy: Optional[Union[dict, Batch]] = {},
            weight: Optional[float] = None,
            **kwargs) -> None:
        """Add a batch of data into replay buffer."""
        if weight is None:
            weight = self._max_prio
        else:
            weight = np.abs(weight)
            self._max_prio = max(self._max_prio, weight)
            self._min_prio = min(self._min_prio, weight)
        self.weight[self._index] = weight ** self._alpha
        super(PrioritizedReplayBuffer, self).add(obs, act, rew, done, obs_next, info, policy)

    def __getitem__(self, index: Union[
        slice, int, np.integer, np.ndarray]) -> Batch:
        obs, obs_next = super()._fetch_obs(index)
        return Batch(
            obs=obs,
            act=self.act[index],
            rew=self.rew[index],
            done=self.done[index],
            obs_next=obs_next,
            info=self.get(index, 'info'),
            policy=self.get(index, 'policy'),
            weight=self.weight[index],
        )
