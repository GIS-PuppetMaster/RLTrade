import tianshou
from typing import Union, Optional
import numpy as np
from tianshou.data import Batch
from collections import OrderedDict
import pandas as pd

from Util.Util import get_submodule


def read_stock_data(stock_codes, stock_data_path, data_type='day', **kwargs):
    stocks = OrderedDict()
    date_index = []
    # in order by stock_code
    stock_codes = [stock_code.replace('.', '_') for stock_code in stock_codes]
    stock_codes = sorted(stock_codes)
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
        self.post_processor = get_submodule(kwargs['post_processor'][0])

    def add(self,
            obs: Union[dict],
            act: Union[np.ndarray, float],
            rew: Union[int, float],
            done: bool,
            obs_next: Optional[Union[dict]] = None,
            info: dict = {},
            policy: Optional[Union[dict, Batch]] = {},
            **kwargs) -> None:
        obs_pass_date = info['obs_pass_date']
        obs_current_date = info['obs_current_date']
        obs_next_pass_date = info['obs_next_pass_date']
        obs_next_current_date = info['obs_next_current_date']
        obs['stock_obs'] = np.array([obs_pass_date,obs_current_date])
        obs_next['stock_obs'] = np.array([obs_next_pass_date, obs_next_current_date])
        super(StockReplayBuffer, self).add(obs, act, rew, done, obs_next, info, policy, **kwargs)

    def _get_stock_obs(self, stock_obs: dict) -> np.ndarray:
        obs_pass_date = stock_obs[0]
        obs_current_date = stock_obs[1]
        pass_index = self.date_list.index(obs_pass_date)
        current_index = self.date_list.index(obs_current_date)
        data = self.value_list[pass_index-1:current_index-1]
        # stack time step
        for i in range(len(data)):
            data[i] = np.expand_dims(data[i], axis=0)
        obs = np.nan_to_num(np.concatenate(data, axis=0))
        obs = self.post_processor(obs.reshape(obs.shape[0], -1)).reshape(obs.shape)
        return obs

    def __getitem__(self, index:np.ndarray) -> Batch:
        obs = self.get(index, 'obs')
        obs_next = self.get(index, 'obs_next')
        obs_replaced = []
        obs_next_replaced = []
        # for each batch
        for i in range(index.shape[0]):
            obs_replaced.append(np.expand_dims(self._get_stock_obs(stock_obs=obs['stock_obs'][i,...]), axis=0))
            obs_next_replaced.append(
                np.expand_dims(self._get_stock_obs(stock_obs=obs_next['stock_obs'][i,...]), axis=0))
        obs_replaced = np.concatenate(obs_replaced, axis=0)
        obs_next_replaced = np.concatenate(obs_next_replaced, axis=0)
        obs['stock_obs'] = obs_replaced
        obs_next['stock_obs'] = obs_next_replaced
        return Batch(
            obs=obs,
            act=self.act[index],
            rew=self.rew[index],
            done=self.done[index],
            obs_next=obs_next,
            info=self.get(index, 'info'),
            policy=self.get(index, 'policy'),
        )
