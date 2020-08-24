from copy import deepcopy

import gym
import pandas as pd
import plotly as py
import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from gym import spaces
from datetime import datetime
from Util.Util import *
import numpy as np
import wandb
import dill
from collections import OrderedDict, defaultdict

"""
日间择时，开盘或收盘交易
"""


# noinspection PyAttributeOutsideInit
class TradeEnv(gym.Env):
    def __init__(self, stock_data_path, start_episode=0, episode_len=720, obs_time_size=60,
                 sim_delta_time=1, stock_codes='000938_XSHE',
                 result_path="E:/运行结果/train/", principal=1e7, poundage_rate=5e-3,
                 time_format="%Y-%m-%d", auto_open_result=False, reward_verbose=1,
                 post_processor=None, trade_time='open', mode='test',
                 agent_state=True, data_type='day', feature_num=32, noise_rate=0., load_from_cache=False):
        """
                :param start_episode: 起始episode
                :param episode_len: episode长度
                :param sim_delta_time: 最小交易频率('x min')
                :param stock_codes: 股票代码
                :param stock_data_path: 数据路径
                :param result_path: 绘图结果保存路径
                :param principal: 初始资金
                :param poundage_rate: 手续费率
                :param time_format: 数据时间格式 str
                :param auto_open_result: 是否自动打开结果
                :param reward_verbose: 0,1,2 不绘制reward，绘制单个reward（覆盖），绘制所有episode的reward
                :param trade_time: 交易时间，open/close
                :param mode: 环境模式，train/test, train模式下会使用wandb记录日志
                :param agent_state: 是否添加agent状态（资金、头寸）到环境状态中
                :param data_type: 数据类型，日级/分钟级，取值：'day'/'minute'
                :param feature_num: 特征数目
                :param 噪声比率， 定义为原始价格数据+标准正态分布噪声*（数据每列方差*noise_rate），也就是加上mean=0，std=noise_rate倍数据std的噪声
                        若为0.则不添加噪声
                :return:
                """
        super(TradeEnv, self).__init__()
        self.delta_time = sim_delta_time
        self.stock_data_path = stock_data_path
        self.result_path = result_path
        self.principal = principal
        self.poundage_rate = poundage_rate
        self.time_format = time_format
        self.auto_open_result = auto_open_result
        self.episode = start_episode
        self.episode_len = episode_len
        self.data_type = data_type
        self.feature_num = feature_num
        self.noise_rate = noise_rate
        self.obs_time = obs_time_size
        self.post_processor = post_processor
        self.agent_state = agent_state
        self.load_from_cache = load_from_cache
        # raw_time_list包含了原始数据中的所有日期
        self.stock_codes, self.stock_data, self.norm_stock_data, self.raw_time_list = self.read_stock_data(stock_codes,
                                                                                                           load_from_cache)
        # time_list只包含交易环境可用的有效日期
        self.time_list = self.raw_time_list[self.obs_time:]
        self.reward_verbose = reward_verbose
        self.action_space = spaces.Box(low=np.array([0, ] + [0 for _ in range(len(self.stock_codes))]),
                                       high=np.array([0, ] + [1 for _ in range(len(self.stock_codes))]))
        if agent_state:
            self.observation_space = spaces.Box(
                low=np.array(
                    [float('-inf') for _ in
                     range(
                         len(self.stock_codes) * self.feature_num * self.obs_time + 1 + len(self.stock_data))]),
                high=np.array(
                    [float('inf') for _ in
                     range(
                         len(self.stock_codes) * self.feature_num * self.obs_time + 1 + len(self.stock_data))]))
        else:
            self.observation_space = spaces.Box(
                low=np.array(
                    [float('-inf') for _ in
                     range(len(self.stock_codes) * self.feature_num * self.obs_time)]),
                high=np.array(
                    [float('inf') for _ in
                     range(len(self.stock_codes) * self.feature_num * self.obs_time)]))
        self.step_ = 0
        assert trade_time == "open" or trade_time == "close"
        self.trade_time = trade_time
        assert mode == "train" or mode == "test" or mode == "eval"
        self.mode = mode
        self.reset()

    def seed(self, seed=None):
        np.random.seed(seed)

    def reset(self):
        # 随机初始化时间
        self.index = np.random.randint(low=0, high=len(self.time_list))
        self.current_time = self.time_list[self.index]
        self.done = False
        self.money = self.principal
        self.buy_value = np.zeros(shape=(len(self.stock_codes, )))
        self.buy_quant = np.zeros(shape=(len(self.stock_codes, )))
        self.weighted_buy_price = np.zeros(shape=(len(self.stock_codes, )))
        self.sold_value = np.zeros(shape=(len(self.stock_codes, )))
        self.sold_quant = np.zeros(shape=(len(self.stock_codes, )))
        self.weighted_sell_price = np.zeros(shape=(len(self.stock_codes, )))
        self.profit_ratio = np.zeros(shape=(len(self.stock_codes, )))
        # 持有股票数目(股)
        self.stock_amount = [0] * len(self.stock_codes)
        # 上次购入股票所花金额
        self.last_time_stock_value = np.zeros(shape=(len(self.stock_codes),))
        # 交易历史
        self.trade_history = []
        self.episode += 1
        self.step_ = 0
        self.start_time = self.current_time
        return self.get_state()

    def step(self, action: np.ndarray):
        action = np.squeeze(action)
        if self.step_ > self.episode_len or self.index >= len(self.time_list):
            self.done = True
        self.step_ += 1
        # 记录交易时间
        trade_time = self.current_time
        # 交易标记
        # traded = [False] * len(self.stock_codes)
        # 当前（分钟）每股收盘/开盘价作为price
        if self.trade_time == 'close':
            price = self.stock_data[self.current_time][:, -1, 1]
        elif self.trade_time == 'open':
            price = self.stock_data[self.current_time][:, -1, 0]
        else:
            raise Exception(f"Wrong trade_time:{self.trade_time}")
        # 停牌股票股价为nan
        nan_mask = np.isnan(price)
        # assert (price[~nan_mask] > 0).all()
        # 此次调整后投入股市的资金
        target_money = self.money * action[-1]
        # 重新计算分配给每只股票的资金数目
        action_masked = deepcopy(action[:-1])
        action_masked[nan_mask] = 0.
        # 把错误分给停牌股票的资金按比例分配给未停牌股票
        action_masked /= action_masked.sum()
        stock_target_money = target_money * action_masked
        # 将因计算精度损失的资金全部加到第一只股票上面
        stock_target_money[0] = target_money - stock_target_money[1:].sum()
        # assert np.abs(stock_target_money.sum() - target_money) < 1e-3
        # 按交易价格计算调整后的持有数量(手)
        target_amount = stock_target_money // (100 * price * (1 + self.poundage_rate))
        # assert (target_amount[~nan_mask] >= 0).all()
        # 当前持有量(手)
        amount = np.array(self.stock_amount)
        # 计算股票数量时将停牌股票数量保持不变
        target_amount[nan_mask] = amount[nan_mask]
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
        # # 收集剩余资金到总体账户(屏蔽停牌部分)
        # self.money += stock_last_money[~nan_mask].sum()
        # assert not np.isnan(target_amount).any()
        self.stock_amount = target_amount
        # 保存交易前所有股票价值
        last_time_value = self.last_time_stock_value.sum()
        # 计算当前每只未停牌股票的价值， 停牌股票价值保留上次计算结果
        self.last_time_stock_value[~nan_mask] = (self.stock_amount * price * 100)[~nan_mask]

        buy_quant = deepcopy(quant)
        buy_quant[buy_quant < 0] = 0.
        buy_price = deepcopy(price)
        buy_price[nan_mask] = 0.
        self.buy_value += buy_price * 100 * buy_quant
        self.buy_quant += buy_quant
        # 加权买价=（历史买入总价值(含当前))/(历史买入股数(含当前)) + 手续费率
        weighted_buy_price = self.buy_value / (100 * self.buy_quant) + self.poundage_rate
        # 不持有股票的结果为nan, 只更新本次买入股票的加权买价
        self.weighted_buy_price[quant > 0] = weighted_buy_price[quant > 0]

        # 卖出量
        sell_quant = deepcopy(quant)
        sell_quant[sell_quant > 0] = 0.
        sell_quant = np.abs(sell_quant)
        sell_price = deepcopy(price)
        sell_price[nan_mask] = 0.
        self.sold_value += sell_price * 100 * sell_quant
        self.sold_quant += sell_quant
        # 加权卖价 = (历史总卖出股票价值(含当前)+当前没卖出的价值)/(历史卖出股票股数(含当前)+当前持有) - 手续费率
        weighted_sell_price = (self.sold_value + self.last_time_stock_value) / (
                100 * (self.sold_quant + self.stock_amount)) - self.poundage_rate
        self.weighted_sell_price = weighted_sell_price
        # self.weighted_sell_price[quant >= 0] = price[quant >= 0]
        # 收益率
        profit_ratio = self.weighted_sell_price / self.weighted_buy_price - 1
        mask = np.logical_and(~np.isnan(profit_ratio), ~np.isinf(profit_ratio))
        self.profit_ratio[mask] = profit_ratio[mask]
        # 计算下一状态和奖励
        # 如果采用t+1结算 and 交易了 则跳到下一天
        self.set_next_day()
        # 先添加到历史中，reward为空
        self.trade_history.append(
            [trade_time, price, quant, deepcopy(self.stock_amount), self.money, None, action,
             deepcopy(self.last_time_stock_value), self.weighted_buy_price, self.weighted_sell_price,
             deepcopy(self.profit_ratio)])
        reward = self.get_reward(last_time_value)
        # 修改历史记录中的reward
        self.trade_history[-1][5] = reward
        return self.get_state(), reward, self.done, {}

    def get_reward(self, last_time_value):
        if len(self.trade_history) >= 2:
            now_hist = self.trade_history[-1]
            now_price = now_hist[1]

            now_value = self.last_time_stock_value.sum() + now_hist[4]
            last_hist = self.trade_history[-2]
            last_price = last_hist[1]
            last_value = last_time_value + last_hist[4]

            price_change_rate = (now_price - last_price) / last_price
            nan_mask = np.isnan(price_change_rate)
            reward = (((now_value - last_value) / last_value) - price_change_rate[~nan_mask].mean()) * 100
        else:
            reward = 0
        return reward

    def set_next_day(self):
        index = self.index
        if index + self.delta_time < len(self.stock_data.keys()):
            self.current_time = self.time_list[index + self.delta_time]
            self.index += self.delta_time
        else:
            self.done = True

    def read_stock_data(self, stock_codes, load_from_cache=False):
        save_path = os.path.join(self.stock_data_path, 'EnvData.dill')
        stocks = OrderedDict()
        date_index = []
        # in order by stock_code
        stock_codes = [stock_code.replace('.', '_') for stock_code in stock_codes]
        stock_codes = sorted(stock_codes)
        if not load_from_cache or not os.path.exists(save_path):
            for idx, stock_code in enumerate(stock_codes):
                print(f'{idx + 1}/{len(stock_codes)} loaded:{stock_code}')
                if self.data_type == 'day':
                    raw = pd.read_csv(self.stock_data_path + stock_code + '_with_indicator.csv', index_col=False)
                else:
                    raise Exception(f"Wrong data type for:{self.data_type}")
                raw_moneyflow = pd.read_csv(self.stock_data_path + stock_code + '_moneyflow.csv', index_col=False)[
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
            # 生成时间序列
            windows_size = self.obs_time
            time_series = OrderedDict()
            post_processed_time_series = OrderedDict()
            # in order by stock_codes
            # 当前时间在state最后
            for i in range(windows_size, len(global_date_intersection)):
                print(f'timeseries: {i + 1 - windows_size}/{len(global_date_intersection) - windows_size}')
                date = global_date_intersection[i]
                stock_data_in_date = []
                post_processed_data = []
                for key in stocks.keys():
                    value = stocks[key][i - windows_size:i, :]
                    stock_data_in_date.append(value.tolist())
                    post_processed_data.append(self.post_processor(value).tolist())
                time_series[date] = np.array(stock_data_in_date)
                post_processed_time_series[date] = np.array(post_processed_data)
            with open(save_path, 'wb') as f:
                dill.dump((stock_codes, time_series, post_processed_time_series, global_date_intersection), f)
        else:
            with open(save_path, 'rb') as f:
                stock_codes_, time_series, post_processed_time_series, global_date_intersection = dill.load(f)
            assert stock_codes == stock_codes_
            assert list(time_series.values())[0].shape == (len(stock_codes), self.obs_time, self.feature_num)
            assert list(time_series.values())[0].shape == list(post_processed_time_series.values())[0].shape
        return stock_codes, time_series, post_processed_time_series, global_date_intersection

    def get_state(self):
        stock_state = self.norm_stock_data[self.current_time]
        stock_state = np.nan_to_num(stock_state)
        state = stock_state.astype(np.float32)
        if self.noise_rate != 0.:
            pass
            # state = np.random.multivariate_normal([0,0,0], [[state.std(axis=0)*self.noise_rate, 0],[0, state.std(axis=1)*self.noise_rate]]) + state
        # state = np.diff(state, axis=0, n=1) / state[1:, :]
        state = state.flatten()
        if self.agent_state:
            state = np.append(state, log10plus1R(self.money + np.array(self.stock_amount)) / 10)
        return state

    def render(self, mode='simple'):
        # if mode == "manual" or self.step_ >= self.episode_len or self.done:
        if mode == 'hybird' or (self.step_ != 0 and self.step_ % 20 == 0):
            return self.draw('hybrid')
        else:
            return self.draw(mode)

    def draw(self, mode):
        if self.trade_history.__len__() <= 1:
            return
        raw_time_array = np.array([i[0] for i in self.trade_history])
        time_list = raw_time_array.tolist()
        # np.ndarray, shape=(time, stock), value = profit for each stock in trade time
        # stock_value_array = np.array([i[7] for i in self.trade_history])

        raw_profit_array = np.array([i[10] for i in self.trade_history])

        # stock_amount = np.array([i[3] for i in self.trade_history])
        # # 每一时刻的每只股票的持股比率,shape=(time, stock)
        # stock_amount_ratio = stock_amount / np.repeat(np.expand_dims(stock_amount.sum(axis=1), axis=1),
        #                                               repeats=stock_amount.shape[1], axis=1)
        # # 每一时刻总资金，shape=(time, )
        # money = np.array([i[4] for i in self.trade_history])
        # # 按持股比例加权的每股等效资金
        # money_weighted = np.repeat(np.expand_dims(money, axis=1), repeats=stock_amount_ratio.shape[1],
        #                            axis=1) * stock_amount_ratio
        # # 按持股比例加权的每股等效本金
        # principal = np.array([self.principal] * len(self.stock_codes))
        # principal_weighted = np.repeat(np.expand_dims(principal, axis=0), repeats=stock_amount_ratio.shape[0],
        #                                axis=0) * stock_amount_ratio
        # raw_profit_array = (stock_value_array + money_weighted - principal_weighted) / principal_weighted
        raw_price_array = pd.DataFrame(np.array([i[1] for i in self.trade_history]).astype(np.float32))
        raw_price_array.fillna(method='ffill', inplace=True)
        raw_price_array = np.array(raw_price_array)
        raw_quant_array = np.array([i[2] for i in self.trade_history]).astype(np.float32)
        raw_amount_array = np.array([i[3] for i in self.trade_history]).astype(np.float32)
        raw_reward_array = np.array([i[5] for i in self.trade_history]).astype(np.float32)
        raw_base_array = (raw_price_array - raw_price_array[0, :]) / raw_price_array[0, :]
        base_nan_mask = np.isnan(raw_base_array)
        base_array = raw_base_array[~base_nan_mask].reshape((raw_base_array.shape[0], -1))
        dis = self.result_path
        path = dis + ("episode_" + str(self.episode - 1) + ".html").replace(':', "_")
        profit_mean = np.mean(raw_profit_array, axis=1).tolist()
        profit_min = np.min(raw_profit_array, axis=1).tolist()[::-1]
        profit_max = np.max(raw_profit_array, axis=1).tolist()

        base_mean = np.mean(base_array, axis=1).tolist()
        base_min = np.min(base_array, axis=1).tolist()[::-1]
        base_max = np.max(base_array, axis=1).tolist()
        if not os.path.exists(dis):
            os.makedirs(dis)
        if mode == 'hybrid':
            fig = make_subplots(rows=2, cols=2, subplot_titles=('回测详情', '奖励', '交易量'),
                                specs=[[{"secondary_y": True}, {"secondary_y": False}],
                                       [{"secondary_y": False}, {"secondary_y": True}]], horizontal_spacing=0.07,
                                vertical_spacing=0.07)
            fig.update_layout(dict(title="回测结果" + "     初始资金：" + str(
                self.principal), paper_bgcolor='#000000', plot_bgcolor='#000000'))
            fig.update_layout(dict(
                xaxis=dict(title='日期', type="date", showgrid=False, zeroline=False),
                xaxis2=dict(title='训练次数', showgrid=False, zeroline=False, titlefont={'color': 'white'},
                            tickfont={'color': 'white'}),
                xaxis3=dict(title='日期', type="date", showgrid=False, zeroline=False),
                xaxis4=dict(title='日期', type="date", showgrid=False, zeroline=False),
                yaxis=dict(title='收益率', showgrid=False, zeroline=False, titlefont={'color': 'red'},
                           tickfont={'color': 'red'}, anchor='x'),
                yaxis2=dict(title='持股量', side='right',
                            titlefont={'color': '#00ccff'}, tickfont={'color': '#00ccff'},
                            showgrid=False, zeroline=False, anchor='x', overlaying='y'),
                yaxis3=dict(title='reward', showgrid=False, zeroline=False, titlefont={'color': '#41AB5D'},
                            tickfont={'color': '#41AB5D'}, anchor='x2', side='left'),
                yaxis4=dict(title='交易量', side='left',
                            titlefont={'color': '#000099'}, tickfont={'color': '#000099'},
                            showgrid=False, zeroline=False, anchor='x3'),
                yaxis5=dict(title='股价', side='left',
                            titlefont={'color': 'orange'}, tickfont={'color': 'orange'},
                            showgrid=False,
                            zeroline=False, anchor='x4'),
                yaxis6=dict(title='收益率', showgrid=False, zeroline=False, titlefont={'color': 'red'},
                            tickfont={'color': 'red'}, anchor='x4', side='right', overlaying='y5'),
                margin=dict(r=10)
            ))
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
                                    line=dict(color='blue'),
                                    mode='lines',
                                    xaxis='x',
                                    yaxis='y')
                price_scatter = dict(x=time_list,
                                     y=price_list,
                                     name=f'股价',
                                     line=dict(color='orange'),
                                     mode='lines',
                                     opacity=1, xaxis='x4',
                                     yaxis='y5')
                trade_bar = dict(x=time_list,
                                 y=quant_list,
                                 name=f'交易量(手)',
                                 marker=dict(color=['red' if quant > 0 else 'green' for quant in quant_list]),
                                 opacity=0.5, xaxis='x3',
                                 yaxis='y4')
                amount_scatter = dict(x=time_list,
                                      y=amount_list,
                                      name=f'持股数量（手）',
                                      line=dict(color='rgba(0,204,255,0.4)'),
                                      mode='lines',
                                      fill='tozeroy',
                                      fillcolor='rgba(0,204,255,0.2)',
                                      opacity=0.6, xaxis='x',
                                      yaxis='y2', secondary_y=True)
                vis = True if i == 0 else False
                for scatter in [profit_scatter, base_scatter, amount_scatter]:
                    fig.add_scatter(**scatter, row=1, col=1, visible=vis)
                fig.add_bar(**trade_bar, row=2, col=1, visible=vis)
                fig.add_scatter(**price_scatter, row=2, col=2, visible=vis)

            fig.add_scatter(**dict(x=time_list,
                                   y=profit_mean,
                                   line=dict(color='rgb(255,0,0)'),
                                   name='profit mean',
                                   showlegend=True), row=2, col=2, visible=True, xaxis='x4', yaxis='y5',
                            secondary_y=True)
            fig.add_scatter(**dict(x=time_list + time_list[::-1],
                                   y=profit_max + profit_min,
                                   fill='toself',
                                   fillcolor='rgba(255,0,0,0.2)',
                                   line_color='rgba(255,255,255,0.1)',
                                   name='profit',
                                   showlegend=False), row=2, col=2, visible=True, xaxis='x4', yaxis='y5',
                            secondary_y=True)
            fig.add_scatter(**dict(x=time_list,
                                   y=base_mean,
                                   line=dict(color='rgb(0,0,255)'),
                                   name='buy and hold mean',
                                   showlegend=True), row=2, col=2, visible=True, xaxis='x4', yaxis='y5',
                            secondary_y=True)
            fig.add_scatter(**dict(x=time_list + time_list[::-1],
                                   y=base_max + base_min,
                                   fill='toself',
                                   fillcolor='rgba(0,0,255,0.2)',
                                   line_color='rgba(255,255,255,0.1)',
                                   name='buy and hold',
                                   showlegend=False), row=2, col=2, visible=True, xaxis='x4', yaxis='y5',
                            secondary_y=True)
            reward_list = raw_reward_array.tolist()
            reward_scatter = dict(x=[i for i in range(len(reward_list))],
                                  y=reward_list,
                                  name='reward',
                                  line=dict(color='#41AB5D'),
                                  mode='lines',
                                  opacity=1, xaxis='x2',
                                  yaxis='y3')
            fig.add_scatter(**reward_scatter, row=1, col=2, visible=True)
            steps = []
            for i in range(0, len(self.stock_codes) * 5, 5):
                step = dict(
                    method="update",
                    args=[{'visible': [False] * (len(self.stock_codes) * 5 + 5)},
                          {'title': f"{self.stock_codes[i // 5]}回测结果, 初始资金：{self.principal}"}],
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
                step['args'][0]['visible'][-4] = True
                step['args'][0]['visible'][-5] = True
                steps.append(step)
            sliders = [dict(
                active=0,
                currentvalue={'prefix': 'StockCoder: '},
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
            fig.add_scatter(**dict(x=time_list + time_list[::-1],
                                   y=profit_max + profit_min,
                                   fill='toself',
                                   fillcolor='rgba(255,0,0,0.2)',
                                   line_color='rgba(255,255,255,0)',
                                   name='profit',
                                   showlegend=True), row=1, col=1, visible=True)
            fig.add_scatter(**dict(x=time_list,
                                   y=base_mean,
                                   line=dict(color='rgb(0,0,255)'),
                                   name='buy and hold mean',
                                   showlegend=True), row=1, col=1, visible=True)
            fig.add_scatter(**dict(x=time_list + time_list[::-1],
                                   y=base_max + base_min,
                                   fill='toself',
                                   fillcolor='rgba(0,0,255,0.2)',
                                   line_color='rgba(255,255,255,0)',
                                   name='buy and hold',
                                   showlegend=True), row=1, col=1, visible=True)
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
        fig.show()
        return raw_profit_array, raw_base_array
