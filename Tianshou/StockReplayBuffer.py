import tianshou
import torch
from typing import Union, Optional, Tuple, Any
import numpy as np
from tianshou.data import Batch, SegmentTree, to_numpy
from collections import OrderedDict
import pandas as pd
import os
from Util.Util import get_modules
import dill

def read_stock_data(stock_codes, stock_data_path, data_type='day', load_from_cache=True, **kwargs):
    # in order by stock_code
    save_path = os.path.join(stock_data_path, 'TradeEnvData.dill')
    stocks = OrderedDict()
    date_index = []
    # in order by stock_code
    stock_codes = [stock_code.replace('.', '_') for stock_code in stock_codes]
    stock_codes = sorted(stock_codes)
    if load_from_cache and os.path.exists(save_path):
        with open(save_path, 'rb') as f:
            stock_codes_, time_series, global_date_intersection = dill.load(f)
        assert stock_codes == stock_codes_
        assert list(time_series.values())[0].shape == (len(stock_codes), kwargs['feature_num'])
        print("buffer数据读取完毕")
    else:
        for idx, stock_code in enumerate(stock_codes):
            print(f'{idx + 1}/{len(stock_codes)} loaded:{stock_code}')
            if data_type == 'day':
                raw = pd.read_csv(stock_data_path + stock_code + '_with_indicator.csv', index_col=False)
            else:
                raise Exception(f"Wrong data type for:{data_type}")
            raw_moneyflow = pd.read_csv(stock_data_path + stock_code + '_moneyflow.csv', index_col=False)[
                ['date', 'change_pct', 'net_pct_main', 'net_pct_xl', 'net_pct_l', 'net_pct_m', 'net_pct_s']].apply(
                lambda x: x / 100 if isinstance(x[1], np.float64) else x)
            raw = pd.merge(raw, raw_moneyflow, left_on='Unnamed: 0', right_on='date', sort=False, copy=False).drop(
                'date', 1).rename(columns={'Unnamed: 0': 'date'})
            raw.fillna(method='ffill', inplace=True)
            date_index.append(np.array(raw['date']))
            raw.set_index('date', inplace=True)
            stocks[stock_code] = raw
        # 生成各支股票数据按时间的并集
        global_date_intersection = date_index[0]
        for i in range(1, len(date_index)):
            global_date_intersection = np.union1d(global_date_intersection, date_index[i])
        global_date_intersection = global_date_intersection.tolist()
        # 根据并集补全停牌数据
        for key in stocks.keys():
            value = stocks[key]
            date = value.index.tolist()
            # 需要填充的日期
            fill = np.setdiff1d(global_date_intersection, date)
            for fill_date in fill:
                value.loc[fill_date] = [np.nan] * value.shape[1]
            if len(fill) > 0:
                value.sort_index(inplace=True)
            stocks[key] = np.array(value)
        # 生成股票数据
        time_series = OrderedDict()
        # in order by stock_codes
        # 当前时间在state最后
        for i in range(len(global_date_intersection)):
            date = global_date_intersection[i]
            stock_data_in_date = []
            for key in stocks.keys():
                value = stocks[key][i, :]
                stock_data_in_date.append(value.tolist())
            time_series[date] = np.array(stock_data_in_date)
    return stock_codes, time_series, global_date_intersection


class StockReplayBuffer(tianshou.data.ReplayBuffer):
    def __init__(self, size: int, **kwargs):
        super().__init__(size, **kwargs)
        assert 'stock_codes' in kwargs.keys()
        assert 'stock_data_path' in kwargs.keys()
        assert 'data_type' in kwargs.keys()
        assert 'post_processor' in kwargs.keys()
        _, stock_data, _ = read_stock_data(**kwargs)

        assert isinstance(stock_data, OrderedDict)
        self.date_list = list(stock_data.keys())
        self.value_list = list(stock_data.values())
        self.post_processor = get_modules(kwargs['post_processor'], 0)

    def add(self,
            obs: dict,
            act: Union[np.ndarray, float],
            rew: Union[int, float],
            done: bool,
            obs_next: Optional[dict] = None,
            info: dict = {},
            policy: Optional[Union[dict, Batch]] = {},
            **kwargs) -> None:
        obs, obs_next = self._convert_obs(obs, obs_next, info)
        super(StockReplayBuffer, self).add(obs, act, rew, done, obs_next, info, policy, **kwargs)

    def _get_stock_obs(self, stock_obs: dict) -> np.ndarray:
        obs_pass_date = stock_obs[0]
        obs_current_date = stock_obs[1]
        pass_index = self.date_list.index(obs_pass_date)
        current_index = self.date_list.index(obs_current_date)
        data = self.value_list[pass_index - 1:current_index - 1]
        # stack time step
        for i in range(len(data)):
            data[i] = np.expand_dims(data[i], axis=0)
        obs = np.nan_to_num(np.concatenate(data, axis=0))
        obs = self.post_processor(obs.reshape(obs.shape[0], -1)).reshape(obs.shape)
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
        obs_replaced = []
        obs_next_replaced = []
        # for each batch
        for i in range(index.shape[0]):
            obs_replaced.append(np.expand_dims(self._get_stock_obs(stock_obs=obs['stock_obs'][i, ...]), axis=0))
            obs_next_replaced.append(
                np.expand_dims(self._get_stock_obs(stock_obs=obs_next['stock_obs'][i, ...]), axis=0))
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


class StockPrioritizedReplayBuffer(StockReplayBuffer):
    """Implementation of Prioritized Experience Replay. arXiv:1511.05952

    :param float alpha: the prioritization exponent.
    :param float beta: the importance sample soft coefficient.

    .. seealso::

        Please refer to :class:`~tianshou.data.ReplayBuffer` for more detailed
        explanation.
    """

    def __init__(self, size: int, alpha: float, beta: float, **kwargs) -> None:
        super().__init__(size, **kwargs)
        assert alpha > 0. and beta >= 0.
        self._alpha, self._beta = alpha, beta
        self._max_prio = 1.
        self._min_prio = 1.
        # bypass the check
        self._weight = SegmentTree(size)
        self.__eps = np.finfo(np.float32).eps.item()

    def __getattr__(self, key: str) -> Union['Batch', Any]:
        """Return self.key"""
        if key == 'weight':
            return self._weight
        return super().__getattr__(key)

    def add(self,
            obs: Union[dict, np.ndarray],
            act: Union[np.ndarray, float],
            rew: Union[int, float],
            done: bool,
            obs_next: Optional[Union[dict, np.ndarray]] = None,
            info: dict = {},
            policy: Optional[Union[dict, Batch]] = {},
            weight: float = None,
            **kwargs) -> None:
        """Add a batch of data into replay buffer."""
        if weight is None:
            weight = self._max_prio
        else:
            weight = np.abs(weight)
            self._max_prio = max(self._max_prio, weight)
            self._min_prio = min(self._min_prio, weight)
        self.weight[self._index] = weight ** self._alpha
        super().add(obs, act, rew, done, obs_next, info, policy)

    def sample(self, batch_size: int) -> Tuple[Batch, np.ndarray]:
        """Get a random sample from buffer with priority probability. Return
        all the data in the buffer if batch_size is ``0``.

        :return: Sample data and its corresponding index inside the buffer.

        The ``weight`` in the returned Batch is the weight on loss function
        to de-bias the sampling process (some transition tuples are sampled
        more often so their losses are weighted less).
        """
        assert self._size > 0, 'Cannot sample a buffer with 0 size!'
        if batch_size == 0:
            indice = np.concatenate([
                np.arange(self._index, self._size),
                np.arange(0, self._index),
            ])
        else:
            scalar = np.random.rand(batch_size) * self.weight.reduce()
            indice = self.weight.get_prefix_sum_idx(scalar)
        batch = self[indice]
        # impt_weight
        # original formula: ((p_j/p_sum*N)**(-beta))/((p_min/p_sum*N)**(-beta))
        # simplified formula: (p_j/p_min)**(-beta)
        batch.weight = (batch.weight / self._min_prio) ** (-self._beta)
        return batch, indice

    def update_weight(self, indice: Union[np.ndarray],
                      new_weight: Union[np.ndarray, torch.Tensor]) -> None:
        """Update priority weight by indice in this buffer.

        :param np.ndarray indice: indice you want to update weight.
        :param np.ndarray new_weight: new priority weight you want to update.
        """
        weight = np.abs(to_numpy(new_weight)) + self.__eps
        self.weight[indice] = weight ** self._alpha
        self._max_prio = max(self._max_prio, weight.max())
        self._min_prio = min(self._min_prio, weight.min())

    def __getitem__(self, index: np.ndarray) -> Batch:
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
