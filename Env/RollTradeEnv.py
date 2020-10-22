import gym
import pandas as pd
import plotly as py
from plotly.subplots import make_subplots
from gym import spaces
from datetime import datetime
from Util.Util import *
import numpy as np
import wandb
import shutil
import dill
from collections import OrderedDict
from empyrical import sortino_ratio
import cupy as cp
from copy import deepcopy
import random
import itertools
import math

"""
日间择时，开盘或收盘交易
"""


class TradeEnv(gym.Env):
    def __init__(self, stock_data_path, config, start_episode=0, episode_len=720, obs_time_size=60,
                 sim_delta_time=1, stock_codes=None,
                 result_path="E:/运行结果/train/", principal=1e7, poundage_rate=5e-3,
                 time_format="%Y-%m-%d", auto_open_result=False,
                 post_processor=None, trade_time='open',
                 agent_state=True, data_type='day', noise_rate=0., load_from_cache=True,
                 wandb_log=False,
                 env_id=0, env_type='test', select_num=20, action_bound=[0, 'inf'], shuffle=False, block_feature=None, **kwargs):
        """
                :param start_episode: 起始episode
                :param episode_len: episode长度
                :param sim_delta_time: 最小交易频率('x min')
                :param stock_codes: 股票代码
                :param stock_data_path: 数据路径，无效，强制读取raw
                :param result_path: 绘图结果保存路径
                :param post_processor: 状态后处理模块，负责标准化状态，[[]],[后处理stock_obs，后处理stock_position，后处理money],
                                        post_processor[i][0]为模块绝对路径，递归导入后面部分
                :param principal: 初始资金
                :param poundage_rate: 手续费率
                :param time_format: 数据时间格式 str
                :param auto_open_result: 是否自动打开结果
                :param trade_time: 交易时间，open/close
                :param agent_state: 是否添加agent状态（资金、头寸）到环境状态中
                :param data_type: 数据类型，日级/分钟级，取值：'day'/'minute'
                :param noise_rate:噪声比率， 定义为原始价格数据+标准正态分布噪声*（数据每列方差*noise_rate），也就是加上mean=0，std=noise_rate倍数据std的噪声
                        若为0.则不添加噪声
                :param env_index:环境编号，用于在Vector_Env中标识环境，
                :return:
                """
        super(TradeEnv, self).__init__()
        assert stock_codes is not None
        self.delta_time = sim_delta_time
        self.stock_data_path = '../Data/raw/'
        self.move_roll_window_step = kwargs['move_roll_window_step']
        self.train_set_days = kwargs['train_set_days']
        self.test_set_days = kwargs['test_set_days']
        self.result_path = result_path
        self.select_num = select_num
        self.shuffle = shuffle
        self.block_feature = block_feature
        if env_id == 0:
            if os.path.exists(self.result_path):
                shutil.rmtree(self.result_path)
            os.makedirs(self.result_path)
        self.principal = principal
        self.poundage_rate = poundage_rate
        self.time_format = time_format
        self.auto_open_result = auto_open_result
        self.episode = start_episode
        self.episode_len = episode_len
        self.data_type = data_type
        self.noise_rate = noise_rate
        post_processor_ = list(itertools.chain.from_iterable(post_processor))
        if not agent_state:
            post_processor_ = post_processor_[0:2]
        if 'diff' in post_processor_:
            obs_time_size += 1
        self.obs_time = obs_time_size
        self.post_processor = get_modules(post_processor)
        self.agent_state = agent_state
        self.load_from_cache = load_from_cache
        # raw_time_list包含了原始数据中的所有日期
        self.stock_codes, self.stock_data, self.stock_data_for_state, self.raw_time_list, data_shape = read_stock_data(stock_data_path, stock_codes, self.post_processor, data_type, obs_time_size,
                                                                                                                       load_from_cache, block_feature=block_feature)
        self.stock_index = np.arange(len(self.stock_codes))
        self.feature_num = data_shape[-1]
        self.obs_stock_num = data_shape[0]
        self.split_point = self.train_set_days
        # time_list只包含交易环境可用的有效日期
        self.time_list = self.raw_time_list[self.obs_time:]
        assert self.move_roll_window(0)
        self.action_space = spaces.Box(low=np.array([action_bound[0] for _ in range(self.obs_stock_num)] + [0, ]),
                                       high=np.array(
                                           [float(action_bound[1]) for _ in range(self.obs_stock_num)] + [
                                               float(action_bound[1]), ]))
        if 'diff' in post_processor_:
            obs_low = np.full((self.obs_time - 1, self.obs_stock_num, self.feature_num), float('-inf'))
            obs_high = np.full((self.obs_time - 1, self.obs_stock_num, self.feature_num), float('inf'))
        elif 'wavelet' in post_processor_:
            obs_low = np.full((math.ceil(self.obs_time / pow(2, 3)), self.obs_stock_num, self.feature_num), float('-inf'))
            obs_high = np.full((math.ceil(self.obs_time / pow(2, 3)), self.obs_stock_num, self.feature_num), float('inf'))
        else:
            obs_low = np.full((self.obs_time, self.obs_stock_num, self.feature_num), float('-inf'))
            obs_high = np.full((self.obs_time, self.obs_stock_num, self.feature_num), float('inf'))
        if agent_state:
            assert len(self.post_processor) == 3
            position_low = np.zeros(shape=(2, self.obs_stock_num))
            position_high = np.full((2, self.obs_stock_num), float('inf'))
            money_low = np.zeros(shape=(1,))
            money_high = np.full((1,), float('inf'))
            self.observation_space = spaces.Dict({'stock_obs': spaces.Box(low=obs_low, high=obs_high),
                                                  'stock_position': spaces.Box(low=position_low, high=position_high),
                                                  'money': spaces.Box(low=money_low, high=money_high)})
        else:
            self.observation_space = spaces.Box(low=obs_low, high=obs_high)
        self.step_ = 0
        assert trade_time == "open" or trade_time == "close"
        self.trade_time = trade_time
        self.env_id = env_id
        self._env_type = env_type
        self.wandb_log = wandb_log
        self.trade_history = []
        if self.noise_rate != 0:
            self.noise_list = [np.random.random((self.obs_time, self.obs_stock_num, self.feature_num)) for _ in range(10000)]
        if wandb_log:
            if config['global_wandb']:
                wandb.init(project=config['wandb']['project'])
        self.init_account()

    def seed(self, seed=None):
        np.random.seed(seed)

    @property
    def env_type(self):
        return self._env_type

    @env_type.setter
    def env_type(self, env_type):
        assert env_type == 'train' or env_type == 'test'
        self.result_path.replace(self._env_type, env_type)
        self._env_type = env_type
        self.reset()

    def init_account(self):
        self.first_buy_value = np.zeros(shape=(len(self.stock_codes, )))
        self.last_sold_value = np.zeros(shape=(len(self.stock_codes, )))
        self.buy_value = np.zeros(shape=(len(self.stock_codes, )))
        self.sold_value = np.zeros(shape=(len(self.stock_codes, )))
        self.money = self.principal
        # 持有股票数目(股)
        self.stock_amount = [0] * self.obs_stock_num
        # 上次购入股票所花金额
        self.stock_value = np.zeros(shape=(self.obs_stock_num,))
        # 交易历史
        self.trade_history = []

    def reset(self):
        if self._env_type == 'train':
            self.activate_time_list = self.time_list[self.split_point - self.train_set_days:self.split_point]
        else:
            self.activate_time_list = self.time_list[self.split_point:self.split_point + self.test_set_days]
        # 初始化时间
        self.index = 0
        self.current_time = self.activate_time_list[self.index]
        self.done = False
        if self._env_type=='train':
            self.init_account()
        self.episode += 1
        self.step_ = 0
        if self.noise_rate != 0 and self.episode > len(self.noise_list):
            self.noise_list = [np.random.multivariate_normal([0], [[self.noise_rate]], (self.obs_time, self.obs_stock_num, self.feature_num))[..., 0] for _ in range(10000)]
        return self.get_state()

    def move_roll_window(self, offset=None):
        if offset is None:
            offset = self.move_roll_window_step
        split_point = self.split_point + offset
        if split_point + self.test_set_days - 1 >= len(self.time_list):
            return False
        else:
            self.split_point = split_point
        return True

    def get_current_price(self):
        if self.trade_time == 'close':
            price = self.stock_data[self.current_time][:, 1]
        elif self.trade_time == 'open':
            price = self.stock_data[self.current_time][:, 0]
        else:
            raise Exception(f"Wrong trade_time:{self.trade_time}")
        return price

    def step(self, action: np.ndarray):
        assert not np.isnan(action).any()
        self.step_ += 1
        if self.step_ >= self.episode_len or self.index >= len(self.time_list):
            self.done = True
        # 当前（分钟）每股收盘/开盘价作为price
        price = self.get_current_price()
        # 停牌股票股价为nan
        nan_mask = np.isnan(price)
        # 减去Tianshou加上的action_bias
        # action = np.squeeze(action).astype(np.float64) - 10
        action = np.squeeze(action).astype(np.float64)
        # 遮盖停牌股票交易指令
        action_masked = action[:-1].copy()
        action_masked[nan_mask] = 0.
        # 剪裁投入资金比例(sigmoid)
        action[-1] = 1 / (1 + np.exp(action[-1]))
        # 最大的前20只股票权重保留，其余置0
        partition = np.argsort(action_masked)
        empty_mask = partition[:- self.select_num]
        trade_mask = partition[-self.select_num:]
        # 非选中股票平仓
        action_masked[empty_mask] = 0.
        # 选中股票使用softmax加权
        sub_action = action_masked[trade_mask]
        normed_sub_action = sub_action - sub_action.max()
        exp_normed_sub_action = np.exp(normed_sub_action)
        action_masked[trade_mask] = exp_normed_sub_action / exp_normed_sub_action.sum()
        # action_masked = np.clip(action_masked, a_min=0, a_max=float('inf'))
        # action_masked /= action_masked.sum()
        # 记录交易时间
        trade_time = self.current_time
        # assert (price[~nan_mask] > 0).all()
        # 此次调整后投入股市的资金
        target_money = self.money * action[-1]
        # 计算每只股票投入资金
        stock_target_money = target_money * action_masked
        # assert np.abs(stock_target_money.sum() - target_money) < 1e-3
        # 按交易价格计算调整后的持有数量(手)
        target_amount = stock_target_money // (100 * price * (1 + self.poundage_rate))
        # assert (target_amount[~nan_mask] >= 0).all()
        # 当前持有量(手)
        amount = np.array(self.stock_amount)
        # 计算股票数量时将停牌股票数量保持不变
        target_amount[nan_mask] = amount[nan_mask]
        # if self.step_ > 1 and self.step_ % 100 != 0:
        #     target_amount = amount
        # 实际交易多少手
        quant = target_amount - amount
        # assert (quant[nan_mask] == 0).all()
        # traded = (quant != 0).all().tolist()
        # 计算交易每只股票所需资金（不含手续费）:每股价格*100股/手*交易手数 +手续费
        stock_cost_money = price * 100 * quant + abs(price * 100 * quant * self.poundage_rate)
        # assert (stock_last_money[~nan_mask] >= 0).all()
        # 更新money
        self.money -= stock_cost_money[~nan_mask].sum()
        # assert self.money > 0
        # assert not np.isnan(target_amount).any()
        self.stock_amount = target_amount
        # 计算并更新当前每只未停牌股票的价值， 停牌股票价值保留上次计算结果
        self.stock_value[~nan_mask] = (self.stock_amount * price * 100)[~nan_mask]

        buy_quant = quant.copy()
        buy_quant[buy_quant < 0] = 0.
        buy_price = price.copy()
        buy_price[nan_mask] = 0.
        buy_value = buy_price * 100 * buy_quant * (1 + self.poundage_rate)
        self.buy_value += buy_value
        self.first_buy_value[self.first_buy_value == 0] = buy_value[self.first_buy_value == 0]
        # 卖出量
        sell_quant = quant.copy()
        sell_quant[sell_quant > 0] = 0.
        sell_quant = np.abs(sell_quant)
        sell_price = price.copy()
        sell_price[nan_mask] = 0.
        sold_value = sell_price * 100 * (1 - self.poundage_rate) * sell_quant
        self.sold_value += sold_value

        # 历史卖出价值（扣除手续费）+当前价格下持有股票的价值)/历史买入花费（算手续费
        profit_ratio = np.nan_to_num(
            (self.stock_value + self.sold_value - self.buy_value) / self.first_buy_value, nan=0., posinf=0.,
            neginf=0.)
        self.last_sold_value = sold_value
        # 计算下一状态和奖励
        # 如果采用t+1结算 and 交易了 则跳到下一天
        self.set_next_day()
        # 先添加到历史中，reward为空
        action[:-1] = action_masked
        # 计算累计回报率
        cum_return_profit_ratio = (self.money + self.stock_value.sum()) / self.principal - 1
        # 计算每天股价变化率
        price_change_rate = price.mean() / np.nan_to_num(self.trade_history[-1][1], nan=0., posinf=0.,
                                                         neginf=0.).mean() - 1 if len(
            self.trade_history) > 0 else 0.
        his_log = [trade_time, price, quant, self.stock_amount.copy(), self.money, None, action,
                   self.stock_value.copy(), self.buy_value.copy(), self.sold_value.copy(), profit_ratio,
                   cum_return_profit_ratio, price_change_rate]
        self.trade_history.append(his_log)
        reward = self.get_reward()
        self.trade_history[-1][5] = reward
        if self.wandb_log:
            wandb.log({f'{self._env_type}_{self.env_id}_episode': self.episode,
                       f'{self._env_type}_{self.env_id}_step': self.step_,
                       f'{self._env_type}_{self.env_id}_reward': reward}, sync=False)
        info = dict(obs_current_date=trade_time, obs_next_current_date=self.current_time,
                    obs_pass_date=self.raw_time_list[self.raw_time_list.index(trade_time) - self.obs_time],
                    obs_next_pass_date=self.raw_time_list[self.raw_time_list.index(self.current_time) - self.obs_time],
                    stock_index=self.stock_index)
        obs = self.get_state()
        if self.done and ((self._env_type == 'train' and self.episode % 20 == 0) or (
                self._env_type == 'test')):
            self.render('hybrid')
        return obs, reward, self.done, info

    def get_state(self):
        time_index = self.raw_time_list.index(self.current_time)
        time_series = self.raw_time_list[time_index - self.obs_time - 1:time_index - 1]
        stock_obs = np.zeros(shape=(self.obs_time, self.obs_stock_num, self.feature_num))
        for idx, date in enumerate(time_series):
            stock_obs[idx, ...] = self.stock_data_for_state[date]
        if self.noise_rate != 0.:
            not_zero_mask = (stock_obs != 0)
            stock_obs[not_zero_mask] += self.noise_list[random.randint(0, len(self.noise_list) - 1)][not_zero_mask]
        if self.post_processor[0].__name__ in force_apply_in_step:
            stock_obs = self.post_processor[0](stock_obs)
        if self.agent_state:
            obs = {'stock_obs': stock_obs}
            # 当前每只股票的每股成本
            stock_cost = np.expand_dims((self.buy_value - self.sold_value) / (100 * np.array(self.stock_amount)),
                                        axis=0)
            stock_cost = np.nan_to_num(stock_cost, nan=0., posinf=0., neginf=0.)
            # shape = (2, num_stocks)
            stock_position = self.post_processor[1](
                np.concatenate([np.expand_dims(self.stock_amount, axis=0), stock_cost], axis=0))
            obs['stock_position'] = stock_position
            money_obs = self.post_processor[2](np.array([self.money, ]))
            obs['money'] = money_obs
            return obs
        else:
            return stock_obs

    def get_reward(self):
        if len(self.trade_history) >= 2:
            now = (self.trade_history[-1][11] + 1) * self.principal
            now_price = self.trade_history[-1][1]
            next_price = self.get_current_price()
            next_mask = np.isnan(next_price)
            now_mask = np.isnan(now_price)
            value = self.stock_value.copy()
            value[~next_mask] = (self.stock_amount * next_price * 100)[~next_mask]
            next = self.money + value.sum()
            if now != 0:
                # reward = ((next / now - 1) - (next_price[~next_mask].mean() / now_price[~now_mask].mean() - 1)) * 100
                reward = (next / now - 1) * 100
            else:
                reward = 0.
            # noncum_return_profit_ratio = np.diff(np.array([i[11] for i in self.trade_history]), prepend=0)
            # minimum_acceptable_return = np.array([i[12] for i in self.trade_history])
            # reward = sortino_ratio(noncum_return_profit_ratio, minimum_acceptable_return)
            # if np.isnan(reward) or np.isinf(reward):
            #     his_reward = np.array([i[5] for i in self.trade_history[:-1]])
            #     his_reward = his_reward.astype(np.float32)
            #     nan_mask = np.isnan(his_reward)
            #     filtered_his_reward = his_reward[~nan_mask]
            #     reward = filtered_his_reward[-1] if filtered_his_reward.shape[0] > 0 else 0
        else:
            reward = 0.
        return reward

    def set_next_day(self):
        index = self.index
        if index + self.delta_time < len(self.activate_time_list):
            self.current_time = self.activate_time_list[index + self.delta_time]
            self.index += self.delta_time
        else:
            self.done = True

    def render(self, mode='hybrid'):
        # if mode == "manual" or self.step_ >= self.episode_len or self.done:
        if self.done:
            if mode == 'hybrid':
                return self.draw('hybrid')
            else:
                return self.draw(mode)

    def draw(self, mode):
        if self.trade_history.__len__() <= 1 or self.env_id != 0:
            return
        raw_time_array = np.array([i[0] for i in self.trade_history])
        time_list = raw_time_array.tolist()
        raw_profit_array = np.array([i[10] for i in self.trade_history])
        raw_price_array = pd.DataFrame(np.array([i[1] for i in self.trade_history]).astype(np.float32))
        raw_price_array.fillna(method='ffill', inplace=True)
        raw_price_array.fillna(method='bfill', inplace=True)
        raw_price_array = np.nan_to_num(np.array(raw_price_array), nan=0., posinf=0., neginf=0.)
        raw_quant_array = np.array([i[2] for i in self.trade_history]).astype(np.float32)
        raw_amount_array = np.array([i[3] for i in self.trade_history]).astype(np.float32)
        raw_reward_array = np.array([i[5] for i in self.trade_history]).astype(np.float32)
        raw_base_array = raw_price_array / raw_price_array[0, :] - 1
        # base_nan_mask = np.isnan(raw_base_array)
        # base_array = raw_base_array[~base_nan_mask].reshape((raw_base_array.shape[0], -1))
        dis = self.result_path
        if self.env_type == 'train':
            path = dis + (f"episode_{self.episode}_id_{self.env_id}.html").replace(':', "_")
        else:
            path = dis + (f"test_id_{self.env_id}.html").replace(':', "_")
        profit_mean = (np.array([i[7] for i in self.trade_history]).sum(axis=1) + np.array(
            [i[4] for i in self.trade_history])) / self.principal - 1
        # profit_min = np.min(raw_profit_array, axis=1).tolist()[::-1]
        # profit_max = np.max(raw_profit_array, axis=1).tolist()

        base_mean = raw_price_array.mean(axis=1) / raw_price_array[0, :].mean() - 1
        # base_min = np.min(base_array, axis=1).tolist()[::-1]
        # base_max = np.max(base_array, axis=1).tolist()
        if self.env_id == 0 and not os.path.exists(dis):
            os.makedirs(dis)
        if mode == 'hybrid':
            fig = make_subplots(rows=2, cols=2,
                                specs=[[{"secondary_y": True}, {"secondary_y": True}],
                                       [{"secondary_y": False}, {"secondary_y": False}]], horizontal_spacing=0.1,
                                vertical_spacing=0.1,
                                shared_xaxes='all')
            fig.update_layout(dict(title="回测结果" + "     初始资金：" + str(
                self.principal), paper_bgcolor='#000000', plot_bgcolor='#000000'))
            buttons = list([
                dict(count=5,
                     label="1w",
                     step="day",
                     stepmode="backward")
            ])
            delta_time = (datetime.strptime(time_list[-1], self.time_format) - datetime.strptime(time_list[0],
                                                                                                 self.time_format))
            if delta_time.days >= 30:
                buttons += [dict(count=1,
                                 label="1m",
                                 step="month",
                                 stepmode="backward"),
                            dict(count=1,
                                 label="MTD",
                                 step="month",
                                 stepmode="todate")]
            if delta_time.days >= 365:
                buttons += [
                    dict(count=1,
                         label="YTD",
                         step="year",
                         stepmode="todate"),
                    dict(count=1,
                         label="1y",
                         step="year",
                         stepmode="backward"),
                ]
            buttons.append(dict(step="all"))
            detail_layout = dict(
                xaxis=dict(type="date", showgrid=False, zeroline=False,
                           rangeselector=dict(
                               buttons=buttons
                           ),
                           rangeslider=dict(visible=True, thickness=0.07), titlefont={'color': 'white'},
                           tickfont={'color': 'white'}, ),
                yaxis=dict(title='收益率', showgrid=True, zeroline=False, titlefont={'color': 'red'},
                           tickfont={'color': 'red'}, anchor='x'),
                yaxis2=dict(title='持股量(手)', side='right',
                            titlefont={'color': '#00ccff'}, tickfont={'color': '#00ccff'},
                            showgrid=False, zeroline=False, anchor='x', overlaying='y'),

                xaxis2=dict(type="date", showgrid=False, zeroline=False, titlefont={'color': 'white'},
                            tickfont={'color': 'white'}, ),
                yaxis3=dict(title='平均收益率', showgrid=True, zeroline=False, titlefont={'color': 'white'},
                            tickfont={'color': 'white'}, anchor='x2', side='left'),
                yaxis4=dict(title='股价(元/股)', side='right',
                            titlefont={'color': 'orange'}, tickfont={'color': 'orange'},
                            showgrid=False,
                            zeroline=False, anchor='x2'),

                xaxis3=dict(type="date", showgrid=False, zeroline=False, titlefont={'color': 'white'},
                            tickfont={'color': 'white'}, ),
                yaxis5=dict(title='交易量(手)', side='left',
                            titlefont={'color': 'white'}, tickfont={'color': 'white'},
                            showgrid=True, zeroline=False, anchor='x3'),

                xaxis4=dict(type="date", showgrid=False, zeroline=False, titlefont={'color': 'white'},
                            tickfont={'color': 'white'}, ),

                yaxis6=dict(title='仓位分配(手)', showgrid=False, zeroline=False, titlefont={'color': 'white'},
                            tickfont={'color': 'white'}, anchor='x4'),
                margin=dict(r=10)
            )
            fig.update_layout(detail_layout)
            for i, stock_code in enumerate(self.stock_codes):
                profit_list = raw_profit_array[:, i].tolist()
                price_list = raw_price_array[:, i].tolist()
                quant_list = raw_quant_array[:, i].tolist()
                amount_list = raw_amount_array[:, i].tolist()
                base_list = raw_base_array[:, i].tolist()
                profit_scatter = dict(x=time_list,
                                      y=profit_list,
                                      name=f'RL',
                                      line=dict(color='red'),
                                      mode='lines',
                                      xaxis='x',
                                      yaxis='y')
                base_scatter = dict(x=time_list,
                                    y=base_list,
                                    name=f'Buy and hold',
                                    line=dict(color='rgb(68,105,255)'),
                                    mode='lines',
                                    xaxis='x',
                                    yaxis='y')
                amount_scatter = dict(x=time_list,
                                      y=amount_list,
                                      name=f'持股数量',
                                      line=dict(color='rgba(0,204,255,0.4)'),
                                      mode='lines',
                                      fill='tozeroy',
                                      fillcolor='rgba(0,204,255,0.2)',
                                      opacity=0.6, xaxis='x',
                                      yaxis='y2', secondary_y=True)
                trade_bar = dict(x=time_list,
                                 y=quant_list,
                                 name=f'交易量',
                                 marker=dict(color=['#FF1A1A' if quant > 0 else '#62C37C' for quant in quant_list]),
                                 opacity=1, xaxis='x3',
                                 yaxis='y5')
                price_scatter = dict(x=time_list,
                                     y=price_list,
                                     name=f'股价',
                                     line=dict(color='orange'),
                                     mode='lines',
                                     opacity=1, xaxis='x2',
                                     yaxis='y4', secondary_y=True)

                vis = True if i == 0 else False
                for scatter in [profit_scatter, base_scatter, amount_scatter]:
                    fig.add_scatter(**scatter, row=1, col=1, visible=vis)
                fig.add_bar(**trade_bar, row=2, col=1, visible=vis)
                fig.add_scatter(**price_scatter, row=1, col=2, visible=vis)

            fig.add_scatter(**dict(x=time_list,
                                   y=profit_mean,
                                   line=dict(color='rgb(255,0,0)'),
                                   name='profit mean',
                                   showlegend=True), row=1, col=2, visible=True, xaxis='x2', yaxis='y3')
            # fig.add_scatter(**dict(x=time_list + time_list[::-1],
            #                        y=profit_max + profit_min,
            #                        fill='toself',
            #                        fillcolor='rgba(200,0,0,0.2)',
            #                        line_color='rgba(255,255,255,0)',
            #                        name='profit min-max',
            #                        showlegend=True), row=1, col=2, visible=True, xaxis='x2', yaxis='y3')
            fig.add_scatter(**dict(x=time_list,
                                   y=base_mean,
                                   line=dict(color='rgb(68,105,255)'),
                                   name='buy and hold mean',
                                   showlegend=True), row=1, col=2, visible=True, xaxis='x2', yaxis='y3')
            # fig.add_scatter(**dict(x=time_list + time_list[::-1],
            #                        y=base_max + base_min,
            #                        fill='toself',
            #                        fillcolor='rgba(68,105,255,0.2)',
            #                        line_color='rgba(255,255,255,0)',
            #                        name='buy and hold min-max',
            #                        showlegend=True), row=1, col=2, visible=True, xaxis='x2', yaxis='y3')
            fig.add_heatmap(
                **dict(x=time_list, y=self.stock_codes,
                       z=raw_amount_array.T, colorscale=[
                        [0, 'rgb(53,50,155)'],
                        [1 / 1000, 'rgb(126,77,143)'],
                        [1 / 100, 'rgb(193,100,121)'],
                        [1 / 10, 'rgb(246,139,69)'],
                        [1, 'rgb(246,211,70)']],
                       name='仓位',
                       showlegend=True, colorbar=dict(len=0.5, y=0.2),
                       customdata=np.array([i[6] for i in self.trade_history])[:, :-1].T,
                       hovertemplate="x:%{x}\ny:%{y}\n金额占比:%{customdata}\n手数:%{z}<extra></extra>"),
                row=2, col=2, visible=True, xaxis='x4', yaxis='y6')
            steps = []
            for i in range(0, self.obs_stock_num * 5, 5):
                step = dict(
                    method="update",
                    args=[{'visible': [False] * (self.obs_stock_num * 5 + 3)},
                          {'title': f"{self.stock_codes[i // 5]}回测结果, 初始资金：{self.principal}"}
                          ],
                    label=self.stock_codes[i // 5],
                )
                step['args'][0]['visible'][i] = True
                step['args'][0]['visible'][i + 1] = True
                step['args'][0]['visible'][i + 2] = True
                step['args'][0]['visible'][i + 3] = True
                step['args'][0]['visible'][i + 4] = True
                step['args'][0]['visible'][-1] = True
                step['args'][0]['visible'][-2] = True
                step['args'][0]['visible'][-3] = True
                steps.append(step)
            sliders = [dict(
                active=0,
                currentvalue={'prefix': 'StockCode: '},
                pad={'t': 50},
                steps=steps,
            )]
            fig.update_layout(sliders=sliders)
        else:
            fig = make_subplots(rows=2, cols=1, subplot_titles=('回测平均结果', '奖励'))
            fig.update_layout(dict(title="回测结果" + "     初始资金：" + str(
                self.principal), paper_bgcolor='#000000', plot_bgcolor='#000000'))
            fig.update_layout(dict(
                xaxis=dict(title='日期', type="date", showgrid=False, zeroline=False),
                xaxis2=dict(title='训练次数', showgrid=False, zeroline=False, titlefont={'color': 'white'},
                            tickfont={'color': 'white'}),
                yaxis=dict(title='收益率', showgrid=False, zeroline=False, titlefont={'color': 'red'},
                           tickfont={'color': 'red'}, anchor='x'),
                yaxis2=dict(title='reward', side='left',
                            titlefont={'color': '#41AB5D'}, tickfont={'color': '#41AB5D'},
                            showgrid=False,
                            zeroline=False, anchor='x2')))
            fig.add_scatter(**dict(x=time_list,
                                   y=profit_mean,
                                   line=dict(color='rgb(255,0,0)'),
                                   name='profit_mean',
                                   showlegend=True), row=1, col=1, visible=True)
            # fig.add_scatter(**dict(x=time_list + time_list[::-1],
            #                        y=profit_max + profit_min,
            #                        fill='toself',
            #                        fillcolor='rgba(255,0,0,0.2)',
            #                        line_color='rgba(255,255,255,0)',
            #                        name='profit',
            #                        showlegend=True), row=1, col=1, visible=True)
            fig.add_scatter(**dict(x=time_list,
                                   y=base_mean,
                                   line=dict(color='rgb(68,105,255)'),
                                   name='buy and hold mean',
                                   showlegend=True), row=1, col=1, visible=True)
            # fig.add_scatter(**dict(x=time_list + time_list[::-1],
            #                        y=base_max + base_min,
            #                        fill='toself',
            #                        fillcolor='rgba(68,105,255,0.2)',
            #                        line_color='rgba(255,255,255,0)',
            #                        name='buy and hold',
            #                        showlegend=True), row=1, col=1, visible=True)
            reward_list = raw_reward_array.tolist()
            reward_scatter = dict(x=[i for i in range(len(reward_list))],
                                  y=reward_list,
                                  name='reward',
                                  line=dict(color='#41AB5D'),
                                  mode='lines',
                                  opacity=1, xaxis='x2',
                                  yaxis='y2')
            fig.add_scatter(**reward_scatter, row=2, col=1, visible=True)
            fig.update_traces(mode='lines')
        py.offline.plot(fig, auto_open=self.auto_open_result, filename=path)
        return raw_profit_array, raw_base_array
