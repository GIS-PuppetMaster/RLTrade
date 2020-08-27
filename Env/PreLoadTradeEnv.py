import gym
import pandas as pd
import plotly as py
from plotly.subplots import make_subplots
from gym import spaces
from datetime import datetime
from Util.Util import *
import numpy as np
import wandb
import dill
from collections import OrderedDict

"""
日间择时，开盘或收盘交易
"""


class TradeEnv(gym.Env):
    def __init__(self, stock_data_path, start_episode=0, episode_len=720, obs_time_size=60,
                 sim_delta_time=1, stock_codes=None,
                 result_path="E:/运行结果/train/", principal=1e7, poundage_rate=5e-3,
                 time_format="%Y-%m-%d", auto_open_result=False,
                 post_processor=None, trade_time='open',
                 agent_state=True, data_type='day', feature_num=32, noise_rate=0., load_from_cache=True, wandb=False):
        """
                :param start_episode: 起始episode
                :param episode_len: episode长度
                :param sim_delta_time: 最小交易频率('x min')
                :param stock_codes: 股票代码
                :param stock_data_path: 数据路径
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
                :param feature_num: 特征数目
                :param 噪声比率， 定义为原始价格数据+标准正态分布噪声*（数据每列方差*noise_rate），也就是加上mean=0，std=noise_rate倍数据std的噪声
                        若为0.则不添加噪声
                :return:
                """
        super(TradeEnv, self).__init__()
        assert stock_codes is not None
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
        self.post_processor = [get_submodule(submodule) for submodule in post_processor]
        self.agent_state = agent_state
        self.load_from_cache = load_from_cache
        # raw_time_list包含了原始数据中的所有日期
        self.stock_codes, self.stock_data, self.norm_stock_data, self.raw_time_list = self.read_stock_data(stock_codes,
                                                                                                           load_from_cache)
        # time_list只包含交易环境可用的有效日期
        self.time_list = self.raw_time_list[self.obs_time:]
        self.action_space = spaces.Box(low=np.array([0 for _ in range(len(self.stock_codes))] + [0, ]),
                                       high=np.array([1 for _ in range(len(self.stock_codes))] + [1, ]))
        obs_low = np.full((self.obs_time, len(self.stock_codes), self.feature_num), float('-inf'))
        obs_high = np.full((self.obs_time, len(self.stock_codes), self.feature_num), float('inf'))
        if agent_state:
            assert len(self.post_processor) == 3
            position_low = np.zeros(shape=(2, len(self.stock_codes)))
            position_high = np.full((2, len(self.stock_codes)), float('inf'))
            money_low = np.zeros(shape=(1,))
            money_high = np.full((1,), float('inf'))
            self.observation_space = spaces.Dict({'stock_obs': spaces.Box(low=obs_low, high=obs_high),
                                                  'stock_position': spaces.Box(low=position_low, high=position_high),
                                                  'money': spaces.Box(low=money_low, high=money_high)})
        else:
            self.observation_space = spaces.Dict({'stock_obs': spaces.Box(low=obs_low, high=obs_high)})
        self.step_ = 0
        assert trade_time == "open" or trade_time == "close"
        self.trade_time = trade_time
        self.wandb = wandb
        self.reset()

    def seed(self, seed=None):
        np.random.seed(seed)

    def reset(self):
        # 随机初始化时间
        self.index = np.random.randint(low=0, high=len(self.time_list) - self.episode_len)
        self.current_time = self.time_list[self.index]
        self.done = False
        self.money = self.principal
        self.buy_value = np.zeros(shape=(len(self.stock_codes, )))
        self.sold_value = np.zeros(shape=(len(self.stock_codes, )))
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
        # 当前（分钟）每股收盘/开盘价作为price
        if self.trade_time == 'close':
            price = self.stock_data[self.current_time][-1:, :, 1]
        elif self.trade_time == 'open':
            price = self.stock_data[self.current_time][-1:, :, 0]
        else:
            raise Exception(f"Wrong trade_time:{self.trade_time}")
        # 停牌股票股价为nan
        nan_mask = np.isnan(price)
        # assert (price[~nan_mask] > 0).all()
        # 此次调整后投入股市的资金
        target_money = self.money * action[-1]
        # 重新计算分配给每只股票的资金数目
        action_masked = action[:-1].copy()
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
        # assert not np.isnan(target_amount).any()
        self.stock_amount = target_amount
        # 保存交易前所有股票价值
        last_time_value = self.last_time_stock_value.sum()
        # 计算并更新当前每只未停牌股票的价值， 停牌股票价值保留上次计算结果
        self.last_time_stock_value[~nan_mask] = (self.stock_amount * price * 100)[~nan_mask]

        buy_quant = quant.copy()
        buy_quant[buy_quant < 0] = 0.
        buy_price = price.copy()
        buy_price[nan_mask] = 0.
        self.buy_value += buy_price * 100 * buy_quant * (1 + self.poundage_rate)

        # 卖出量
        sell_quant = quant.copy()
        sell_quant[sell_quant > 0] = 0.
        sell_quant = np.abs(sell_quant)
        sell_price = price.copy()
        sell_price[nan_mask] = 0.
        self.sold_value += sell_price * 100 * (1 - self.poundage_rate) * sell_quant

        # q用来计算每只股票当前一共交易了多少次
        if self.step_ == 1:
            q = np.expand_dims(quant, axis=0)
        else:
            q = np.concatenate(
                [np.array([i[2] for i in self.trade_history]).astype(np.float32), np.expand_dims(quant, axis=0)],
                axis=0)
        # ((历史卖出价值（扣除手续费）+当前价格下持有股票的价值)/历史买入花费（算手续费）-1) * 当前交易次数
        # 前面那一坨算的是平均每次交易的收益率，要知道当前的总收益率，只需要乘以当前一共交易了几次
        profit_ratio = ((self.sold_value + self.last_time_stock_value) / self.buy_value - 1) * np.count_nonzero(q,
                                                                                                                axis=0)
        mask = np.logical_or(np.isnan(profit_ratio), np.isinf(profit_ratio))
        profit_ratio[mask] = 0.
        # 计算下一状态和奖励
        # 如果采用t+1结算 and 交易了 则跳到下一天
        self.set_next_day()
        # 先添加到历史中，reward为空
        his_log = [trade_time, price, quant, self.stock_amount.copy(), self.money, None, action,
                   self.last_time_stock_value.copy(), self.buy_value, self.sold_value, profit_ratio]

        reward = self.get_reward(his_log, last_time_value)
        his_log[5] = reward
        self.trade_history.append(his_log)
        if self.wandb:
            wandb.log({'episode': self.episode, 'reward': reward}, sync=False)
        info = dict(obs_current_date=trade_time, obs_next_current_date=self.current_time,
                    obs_pass_date=self.raw_time_list[self.raw_time_list.index(trade_time) - self.obs_time],
                    obs_next_pass_date=self.raw_time_list[self.raw_time_list.index(self.current_time) - self.obs_time])
        return self.get_state(trade_time), reward, self.done, info

    def get_state(self, trade_time):
        stock_obs = self.norm_stock_data[trade_time]
        stock_obs = np.nan_to_num(stock_obs)
        stock_obs = stock_obs.astype(np.float32)
        if self.noise_rate != 0.:
            pass
            # state = np.random.multivariate_normal([0,0,0], [[state.std(axis=0)*self.noise_rate, 0],[0, state.std(axis=1)*self.noise_rate]]) + state
        # state = np.diff(state, axis=0, n=1) / state[1:, :]
        obs = {'stock_obs': stock_obs}
        if self.agent_state:
            # 当前每只股票的每股成本
            stock_cost = np.expand_dims((self.buy_value - self.sold_value) / (100 * np.array(self.stock_amount)),
                                        axis=0)
            stock_cost = np.nan_to_num(stock_cost)
            # shape = (2, num_stocks)
            stock_position = self.post_processor[1](
                np.concatenate([np.expand_dims(self.stock_amount, axis=0), stock_cost], axis=0))
            obs['stock_position'] = stock_position
            money_obs = self.post_processor[2](np.array([self.money, ]))
            obs['money'] = money_obs
        return obs

    def get_reward(self, now_hist, last_time_value):
        if len(self.trade_history) >= 2:
            now_price = now_hist[1]
            now_value = self.last_time_stock_value.sum() + now_hist[4]
            last_hist = self.trade_history[-1]
            last_price = last_hist[1]
            last_value = last_time_value + last_hist[4]

            price_change_rate = (now_price - last_price) / last_price
            nan_mask = np.isnan(price_change_rate)
            # 超额收益率一阶差分*100
            reward = (((now_value - last_value) / last_value) - price_change_rate[~nan_mask].mean()) * 100
            # 累计收益率
            # reward = now_hist[-1].mean()
        else:
            reward = 0
        return reward

    def set_next_day(self):
        index = self.index
        if index + self.delta_time < len(self.time_list):
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
                    post_processed_data.append(self.post_processor[0](value).tolist())
                # stock, time, feature->time, stock, feature
                time_series[date] = np.array(stock_data_in_date).transpose((1, 0, 2))
                post_processed_time_series[date] = np.array(post_processed_data).transpose((1, 0, 2))
            with open(save_path, 'wb') as f:
                dill.dump((stock_codes, time_series, post_processed_time_series, global_date_intersection), f)
            print("数据生成/保存完毕")
        else:
            with open(save_path, 'rb') as f:
                stock_codes_, time_series, post_processed_time_series, global_date_intersection = dill.load(f)
            assert stock_codes == stock_codes_
            assert list(time_series.values())[0].shape == (len(stock_codes), self.obs_time, self.feature_num)
            assert list(time_series.values())[0].shape == list(post_processed_time_series.values())[0].shape
            print("数据读取完毕")
        return stock_codes, time_series, post_processed_time_series, global_date_intersection


    def render(self, mode='simple'):
        # if mode == "manual" or self.step_ >= self.episode_len or self.done:
        if self.done:
            if mode == 'hybird' or (self.step_ != 0 and self.step_ % 20 == 0):
                return self.draw('hybrid')
            else:
                return self.draw(mode)

    def draw(self, mode):
        if self.trade_history.__len__() <= 1:
            return
        raw_time_array = np.array([i[0] for i in self.trade_history])
        time_list = raw_time_array.tolist()
        raw_profit_array = np.array([i[10] for i in self.trade_history])
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
            fig = make_subplots(rows=2, cols=2, subplot_titles=('回测详情', '', '交易量', '股价'),
                                specs=[[{"secondary_y": True}, {"secondary_y": False}],
                                       [{"secondary_y": False}, {"secondary_y": False}]], horizontal_spacing=0.1,
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
                           rangeslider=dict(visible=True, thickness=0.1), titlefont={'color': 'white'},
                           tickfont={'color': 'white'}, ),
                yaxis=dict(title='收益率', showgrid=False, zeroline=False, titlefont={'color': 'red'},
                           tickfont={'color': 'red'}, anchor='x'),
                yaxis2=dict(title='持股量(手)', side='right',
                            titlefont={'color': '#00ccff'}, tickfont={'color': '#00ccff'},
                            showgrid=False, zeroline=False, anchor='x', overlaying='y'),

                xaxis2=dict(type="date", showgrid=False, zeroline=False, titlefont={'color': 'white'},
                            tickfont={'color': 'white'}, ),
                yaxis3=dict(title='平均收益率', showgrid=False, zeroline=False, titlefont={'color': 'white'},
                            tickfont={'color': 'white'}, anchor='x2', side='left'),

                xaxis3=dict(type="date", showgrid=False, zeroline=False, titlefont={'color': 'white'},
                            tickfont={'color': 'white'}, ),
                yaxis4=dict(title='交易量(手)', side='left',
                            titlefont={'color': 'white'}, tickfont={'color': 'white'},
                            showgrid=False, zeroline=False, anchor='x3'),

                xaxis4=dict(type="date", showgrid=False, zeroline=False, titlefont={'color': 'white'},
                            tickfont={'color': 'white'}, ),
                yaxis5=dict(title='股价(元/股)', side='left',
                            titlefont={'color': 'orange'}, tickfont={'color': 'orange'},
                            showgrid=False,
                            zeroline=False, anchor='x4'),
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
                                    line=dict(color='blue'),
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
                                 opacity=0.5, xaxis='x3',
                                 yaxis='y4')
                price_scatter = dict(x=time_list,
                                     y=price_list,
                                     name=f'股价',
                                     line=dict(color='orange'),
                                     mode='lines',
                                     opacity=1, xaxis='x4',
                                     yaxis='y5')

                vis = True if i == 0 else False
                for scatter in [profit_scatter, base_scatter, amount_scatter]:
                    fig.add_scatter(**scatter, row=1, col=1, visible=vis)
                fig.add_bar(**trade_bar, row=2, col=1, visible=vis)
                fig.add_scatter(**price_scatter, row=2, col=2, visible=vis)

            fig.add_scatter(**dict(x=time_list,
                                   y=profit_mean,
                                   line=dict(color='rgb(255,0,0)'),
                                   name='profit mean',
                                   showlegend=True), row=1, col=2, visible=True, xaxis='x2', yaxis='y3')
            fig.add_scatter(**dict(x=time_list + time_list[::-1],
                                   y=profit_max + profit_min,
                                   fill='toself',
                                   fillcolor='rgba(200,0,0,0.2)',
                                   line_color='rgba(255,255,255,0)',
                                   name='profit',
                                   showlegend=True), row=1, col=2, visible=True, xaxis='x2', yaxis='y3')
            fig.add_scatter(**dict(x=time_list,
                                   y=base_mean,
                                   line=dict(color='rgb(68,105,255)'),
                                   name='buy and hold mean',
                                   showlegend=True), row=1, col=2, visible=True, xaxis='x2', yaxis='y3')
            fig.add_scatter(**dict(x=time_list + time_list[::-1],
                                   y=base_max + base_min,
                                   fill='toself',
                                   fillcolor='rgba(68,105,255,0.2)',
                                   line_color='rgba(255,255,255,0)',
                                   name='buy and hold',
                                   showlegend=True), row=1, col=2, visible=True, xaxis='x2', yaxis='y3')
            steps = []
            for i in range(0, len(self.stock_codes) * 5, 5):
                step = dict(
                    method="update",
                    args=[{'visible': [False] * (len(self.stock_codes) * 5 + 4)},
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
                step['args'][0]['visible'][-4] = True
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
                                   line=dict(color='rgb(68,105,255)'),
                                   name='buy and hold mean',
                                   showlegend=True), row=1, col=1, visible=True)
            fig.add_scatter(**dict(x=time_list + time_list[::-1],
                                   y=base_max + base_min,
                                   fill='toself',
                                   fillcolor='rgba(68,105,255,0.2)',
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
        return raw_profit_array, raw_base_array