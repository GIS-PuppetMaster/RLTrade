import gym
import pandas as pd
import plotly as py
import plotly.graph_objs as go
from gym import spaces
from datetime import datetime
from Util.Util import *
import numpy as np

"""
日间择时，开盘或收盘交易
"""


# noinspection PyAttributeOutsideInit
class TradeEnv(gym.Env):
    def __init__(self, stock_data_path, start_episode=0, episode_len=720, obs_time_size='60 day',
                 obs_delta_frequency='1 day',
                 sim_delta_time='1 min', stock_code='000938_XSHE',
                 result_path="E:/运行结果/train/", principal=1e5, origin_stock_amount=0, poundage_rate=5e-3,
                 time_format="%Y-%m-%d", auto_open_result=False, reward_verbose=1,
                 post_processor=None, start_index_bound=None, end_index_bound=None, trade_time='open'):
        """
                :param episode: 起始episode
                :param episode_len: episode长度
                :param sim_delta_time: 最小交易频率('x min')
                :param draw_frequency: render模式的绘制频率(step/次)
                :param stock_code: 股票代码
                :param stock_data_path: 数据路径
                :param result_path: 绘图结果保存路径
                :param principal: 初始资金
                :param gamma: 奖励价格加权衰减率
                :param origin_stock_amount:初始股票股数
                :param poundage_rate: 手续费率
                :param time_format: 数据时间格式 str
                :param t1settlement: A股T+1结算，每天只能交易一次
                :param auto_open_result: 是否自动打开结果
                :param reward_verbose: 0,1,2 不绘制reward，绘制单个reward（覆盖），绘制所有episode的reward
                :param read_data: 是否读入数据
                :return:
                """
        super(TradeEnv, self).__init__()
        self.delta_time_size = sim_delta_time
        self.delta_time = int(sim_delta_time[0:-4])
        self.stock_code = stock_code
        self.stock_data_path = stock_data_path
        self.stock_code = stock_code
        self.result_path = result_path
        self.principal = principal
        self.origin_stock_amount = origin_stock_amount
        self.poundage_rate = poundage_rate
        self.time_format = time_format
        self.auto_open_result = auto_open_result
        self.episode = start_episode
        self.episode_len = episode_len
        self.stock_data, self.keys = self.read_stock_data()
        self.end_time = list(self.stock_data.keys())[-1]
        self.reward_verbose = reward_verbose
        self.obs_time = int(obs_time_size[0:-4])
        self.obs_delta_frequency = int(obs_delta_frequency[0:-4])
        self.action_space = spaces.Box(low=np.array([-1]), high=np.array([1]))
        self.observation_space = spaces.Box(
            low=np.array([[float('-inf') for _ in range(26)]
                          for _ in range(self.obs_time // self.obs_delta_frequency)]),
            high=np.array([[float('inf') for _ in range(26)]
                           for _ in range(0, self.obs_time // self.obs_delta_frequency)]))
        self.step_ = 0
        self.his_reward = []
        self.post_processor = post_processor
        assert trade_time == "open" or trade_time == "close"
        self.trade_time = trade_time
        self.start_index_bound = self.obs_time + 1
        if start_index_bound is not None:
            assert self.start_index_bound <= start_index_bound
            self.start_index_bound = start_index_bound

        if end_index_bound is None:
            self.end_index_bound = len(self.stock_data) - 10
        else:
            self.end_index_bound = end_index_bound

    def seed(self, seed=None):
        np.random.seed(seed)

    def reset(self):
        # 随机初始化时间
        self.current_time = \
            np.random.choice(np.array(list(self.stock_data.keys()))[self.start_index_bound:self.end_index_bound], 1)[0]
        self.index = self.keys.index(self.current_time)
        self.episode_end_index = self.index + self.episode_len if self.index + self.episode_len < len(self.keys) else (
                len(self.keys) - 1)
        self.done = False
        self.money = self.principal
        # 持有股票数目(股)
        self.stock_amount = self.origin_stock_amount
        # 交易历史
        self.trade_history = []
        self.episode += 1
        self.step_ = 0
        self.profit_list = []
        self.next_day_counter = 0
        self.start_time = self.current_time
        return self.get_state()

    def step(self, action):
        if self.step_ >= self.episode_len - 1:
            self.done = True
        self.step_ += 1
        quant = 0
        # 记录交易时间
        temp_time = self.current_time
        # 惩罚标记
        flag = False
        # 交易标记
        traded = False
        # 当前（分钟）每股收盘/开盘价作为price
        if self.trade_time == 'close':
            price = self.stock_data[self.current_time][1]
        elif self.trade_time == 'open':
            price = self.stock_data[self.current_time][0]
        # 买入
        action = np.squeeze(action)
        action = [action]
        if action[0] > 0:
            # 按钱数百分比买入
            # 当前的钱可以买多少手
            amount = self.money // (100 * price * (1 + self.poundage_rate))
            # 实际买多少手
            quant = int(action[0] * amount)
            if quant == 0:
                # print("钱数：" + str(self.money) + "不足以购买一手股票")
                flag = True
            else:
                traded = True
        # 卖出
        elif action[0] < 0:
            # 当前手中有多少手
            amount = self.stock_amount / 100
            if amount == 0:
                flag = True
            else:
                # 实际卖出多少手
                quant = int(action[0] * amount)
                if quant == 0:
                    flag = True
                else:
                    traded = True
        # 记录交易前数据
        old_money = self.money
        old_amount = self.stock_amount
        # 钱数-=每股价格*100*交易手数+手续费
        self.money = self.money - price * 100 * quant - abs(price * 100 * quant * self.poundage_rate)
        # 如果因为action + 随机数导致money<0 则取消交易
        if self.money < 0:
            self.money = old_money
            quant = 0
            traded = False
        else:
            # 股票数
            self.stock_amount += 100 * quant
            # 如果因为action + 随机数导致amount<0 则取消交易
            if self.stock_amount < 0:
                self.money = old_money
                self.stock_amount = old_amount
                quant = 0
                traded = False

        # 计算下一状态和奖励
        # 如果采用t+1结算 and 交易了 则跳到下一天
        self.set_next_day()
        # 先添加到历史中，reward为空
        self.trade_history.append(
            [temp_time, price, quant, self.stock_amount, self.money, None, action[0]])
        reward = self.get_reward()
        # 修改历史记录中的reward
        self.trade_history[-1][5] = reward
        return self.get_state(), reward, self.done, {}

    def get_value(self, last_hist):
        last_value = last_hist[1] * last_hist[3] + last_hist[4]
        return last_value

    def get_reward(self):
        now_hist = self.trade_history[-1]
        now_value = self.get_value(now_hist)
        now_price = now_hist[1]
        if len(self.trade_history) >= 2:
            last_hist = self.trade_history[-2]
            last_value = self.get_value(last_hist)
            last_price = last_hist[1]
        else:
            last_value = self.principal
            if self.trade_time == 'close':
                last_price = self.stock_data[self.start_time][1]
            elif self.trade_time == 'open':
                last_price = self.stock_data[self.start_time][0]
        if last_value == 0:
            last_value = self.principal
        reward = (((now_value - last_value) / last_value) - ((now_price - last_price) / last_price)) * 100
        # if len(self.trade_history) > 10 and (np.array(self.trade_history[-10:])[:,2]==0).all():
        #     reward -= 1

        return reward

    def render(self, mode='auto'):
        # if mode == "manual" or self.step_ >= self.episode_len or self.done:
        if mode == 'manual' or (self.step_ != 0 and self.step_ % 20 == 0):
            return self.draw()

    def get_last_time(self):
        assert self.index - self.delta_time >= 0
        index = self.index - self.delta_time
        return self.keys[index], index

    def set_next_day(self):
        index = self.index
        if index + self.delta_time <= self.keys.__len__() - 1:
            self.current_time = self.keys[index + self.delta_time]
            self.index += self.delta_time
        else:
            self.done = True

    def read_stock_data(self):
        raw = pd.read_csv(self.stock_data_path + self.stock_code + '_with_indicator.csv', index_col=False)
        raw = raw.dropna(axis=1, how='any')
        data = np.array(raw)
        res = {}
        time_list = []
        for i in range(0, data.shape[0]):
            line = data[i, :]
            date = datetime.strptime(str(line[0]), self.time_format)
            res[date] = line[1:]
            time_list.append(date)
        return res, time_list

    def get_state(self):
        stock_state = []
        # 回溯历史股票状态
        # 从上一时刻的价格开始回溯
        time, index = self.get_last_time()
        if time is None:
            self.done = True
            return None
        for _ in range(int(self.obs_time / self.obs_delta_frequency)):
            stock_state.append(self.stock_data[time].tolist())
            index -= self.obs_delta_frequency
            if index >= 0:
                time = self.keys[index]
            else:
                time = self.keys[0]
                self.done = True
        stock_state = np.flip(stock_state, axis=0)
        state = stock_state.astype(np.float32)
        if self.post_processor is not None:
            state = self.post_processor(state)
        return state

    def draw(self):
        if self.trade_history.__len__() <= 1:
            return
        his = np.array(self.trade_history)
        time_list = np.squeeze(his[:, 0]).tolist()
        profit_list = np.squeeze(
            (his[:, 4].astype(np.float32) + his[:, 1].astype(np.float32) * his[:, 3].astype(
                np.float32) - self.principal) / self.principal).tolist()
        price_list = np.squeeze(his[:, 1]).tolist()
        quant_list = np.squeeze(his[:, 2]).tolist()
        amount_list = np.squeeze(his[:, 3]).tolist()
        reward_list = np.squeeze(his[:, 5]).tolist()
        price_array = np.array(price_list).astype(np.float32)
        base_list = ((price_array - price_array[0]) / price_array[0]).tolist()
        dis = self.result_path
        path = dis + ("episode_" + str(self.episode - 1) + ".html").replace(':', "_")
        if not os.path.exists(dis):
            os.makedirs(dis)
        profit_scatter = go.Scatter(x=time_list,
                                    y=profit_list,
                                    name='RL',
                                    line=dict(color='red'),
                                    mode='lines')
        base_scatter = go.Scatter(x=time_list,
                                  y=base_list,
                                  name='Base',
                                  line=dict(color='blue'),
                                  mode='lines')
        price_scatter = go.Scatter(x=time_list,
                                   y=price_list,
                                   name='股价',
                                   line=dict(color='orange'),
                                   mode='lines',
                                   xaxis='x',
                                   yaxis='y2',
                                   opacity=1)
        trade_bar = go.Bar(x=time_list,
                           y=quant_list,
                           name='交易量（手）',
                           marker_color='#000099',
                           xaxis='x',
                           yaxis='y3',
                           opacity=0.5)
        amount_scatter = go.Scatter(x=time_list,
                                    y=amount_list,
                                    name='持股数量（手）',
                                    line=dict(color='rgba(0,204,255,0.4)'),
                                    mode='lines',
                                    fill='tozeroy',
                                    fillcolor='rgba(0,204,255,0.2)',
                                    xaxis='x',
                                    yaxis='y4',
                                    opacity=0.6)
        py.offline.plot({
            "data": [profit_scatter, base_scatter, price_scatter, trade_bar,
                     amount_scatter],
            "layout": go.Layout(
                title=self.stock_code + " 回测结果" + "     初始资金：" + str(
                    self.principal) + "     初始股票总量(股)：" + str(
                    self.origin_stock_amount),
                xaxis=dict(title='日期', type="category", showgrid=False, zeroline=False),
                yaxis=dict(title='收益率', showgrid=False, zeroline=False, titlefont={'color': 'red'},
                           tickfont={'color': 'red'}),
                yaxis2=dict(title='股价', overlaying='y', side='right',
                            titlefont={'color': 'orange'}, tickfont={'color': 'orange'},
                            showgrid=False,
                            zeroline=False),
                yaxis3=dict(title='交易量', overlaying='y', side='right',
                            titlefont={'color': '#000099'}, tickfont={'color': '#000099'},
                            showgrid=False, position=0.97, zeroline=False, anchor='free'),
                yaxis4=dict(title='持股量', overlaying='y', side='left',
                            titlefont={'color': '#00ccff'}, tickfont={'color': '#00ccff'},
                            showgrid=False, position=0.03, zeroline=False, anchor='free'),
                paper_bgcolor='#000000',
                plot_bgcolor='#000000'
            )
        }, auto_open=self.auto_open_result, filename=path)
        if self.reward_verbose != 0:
            reward_scatter = go.Scatter(x=[i for i in range(len(reward_list))],
                                        y=reward_list,
                                        name='reward',
                                        line=dict(color='orange'),
                                        mode='lines',
                                        opacity=1)
            if self.reward_verbose == 1:
                path = dis + "reward.html".format(self.episode - 1)
            else:
                path = dis + "reward_{}.html".format(self.episode - 1)

            py.offline.plot({
                "data": [reward_scatter],
                "layout": go.Layout(
                    title="reward",
                    xaxis=dict(title='训练次数', showgrid=False, zeroline=False, titlefont={'color': 'white'},
                               tickfont={'color': 'white'}),
                    yaxis=dict(title='reward', showgrid=False, zeroline=False, titlefont={'color': 'orange'},
                               tickfont={'color': 'orange'}),
                    paper_bgcolor='#000000',
                    plot_bgcolor='#000000'
                )
            }, auto_open=self.auto_open_result, filename=path)

        return profit_list, base_list
