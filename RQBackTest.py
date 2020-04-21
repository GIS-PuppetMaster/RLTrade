# coding=utf-8
from rqalpha.api import *
from stable_baselines import TRPO
from ta.momentum import *
from ta.volatility import *
from ta.trend import *
from ta.volume import *
from ta.others import *
import pandas as pd
import numpy as np
stock_code = '000938.XSHE'
indicators = [
    ('RSI', rsi, ['close']),
    ('MFI', money_flow_index, ['high', 'low', 'close', 'volume']),
    ('TSI', tsi, ['close']),
    ('UO', uo, ['high', 'low', 'close']),
    ('AO', ao, ['high', 'close']),
    ('MACDDI', macd_diff, ['close']),
    ('VIP', vortex_indicator_pos, ['high', 'low', 'close']),
    ('VIN', vortex_indicator_neg, ['high', 'low', 'close']),
    ('TRIX', trix, ['close']),
    ('MI', mass_index, ['high', 'low']),
    ('CCI', cci, ['high', 'low', 'close']),
    ('DPO', dpo, ['close']),
    # ('KST', kst, ['close']),
    # ('KSTS', kst_sig, ['close']),
    ('ARU', aroon_up, ['close']),
    ('ARD', aroon_down, ['close']),
    # ('ARI', diff, ['ARU', 'ARD']),
    ('BBH', bollinger_hband, ['close']),
    ('BBL', bollinger_lband, ['close']),
    ('BBM', bollinger_mavg, ['close']),
    # ('BBHI', bollinger_hband_indicator, ['close']),
    # ('BBLI', bollinger_lband_indicator, ['close']),
    # ('KCHI', keltner_channel_hband_indicator, ['high', 'low', 'close']),
    # ('KCLI', keltner_channel_lband_indicator, ['high', 'low', 'close']),
    # ('DCHI', donchian_channel_hband_indicator, ['close']),
    # ('DCLI', donchian_channel_lband_indicator, ['close']),
    # ('ADI', acc_dist_index, ['high', 'low', 'close', 'volume']),
    # ('OBV', on_balance_volume, ['close', 'volume']),
    ('CMF', chaikin_money_flow, ['high', 'low', 'close', 'volume']),
    # ('FI', force_index, ['close', 'volume']),
    # ('EM', ease_of_movement, ['high', 'low', 'close', 'volume']),
    # ('VPT', volume_price_trend, ['close', 'volume']),
    # ('NVI', negative_volume_index, ['close', 'volume']),
    ('DR', daily_return, ['close']),
    ('DLR', daily_log_return, ['close'])
]


def get_indicator(raw):
    for indicator_name, fun, col_name in indicators:
        print(indicator_name)
        # 对于参数个数不同的计算因子函数，执行不同的参数传入方法（笨写法，高级的可变参数写法不会）
        if len(col_name) == 1:
            temp = fun(raw[col_name[0]]).to_frame()
            temp.columns = [indicator_name]
        elif len(col_name) == 2:
            temp = fun(raw[col_name[0]], raw[col_name[1]]).to_frame()
            temp.columns = [indicator_name]
        elif len(col_name) == 3:
            temp = fun(raw[col_name[0]], raw[col_name[1]], raw[col_name[2]]).to_frame()
            temp.columns = [indicator_name]
        elif len(col_name) == 4:
            temp = fun(raw[col_name[0]], raw[col_name[1]], raw[col_name[2]], raw[col_name[3]]).to_frame()
            temp.columns = [indicator_name]
        else:
            raise Exception("no support arg num")
        raw = pd.concat([raw, temp], axis=1)
    return raw


def read_csv_as_df(path):
    import pandas as pd
    return pd.read_csv(path)


def init(context):
    import os
    # strategy_file_path = context.config.base.strategy_file
    # csv_path = os.path.join(os.path.dirname(strategy_file_path), "./Data/test/000938_XSHE_day.csv")
    # context.XSHE000938_df = read_csv_as_df(csv_path)
    import os
    strategy_file_path = context.config.base.strategy_file
    net_type = 'large_net'
    file_list = os.listdir(os.path.join(os.path.dirname(strategy_file_path), './checkpoints/'+net_type+'/'))
    max_index = -1
    max_file_name = ''
    for filename in file_list:
        index = int(filename.split("_")[2])
        if index > max_index:
            max_index = index
            max_file_name = filename
    max_file_name = 'rl_model_553984_steps.zip'
    model_path = os.path.join(os.path.dirname(strategy_file_path), "./checkpoints/"+net_type+"/"+max_file_name)
    logger.info('model_path:'+model_path)
    model = TRPO.load(model_path)
    context.model = model
    context.stock_code = stock_code

def before_trading(context):
    pass

def handle_bar(context, bar):
    # 获取datetime, open,high,low,close,money，不包括当天数据
    price = history_bars(context.stock_code, 120, '1d', include_now=False, skip_suspended=True)
    # 剔除datetime
    price = pd.DataFrame(price).values[:, 1:]
    # 获取volume
    volume = history_bars(context.stock_code, 120, '1d', 'total_turnover', include_now=False, skip_suspended=True)
    # 拼接volume：open,high,low,close,volume,money
    price = np.insert(price, 4, volume, axis=1)
    # 调整顺序为open,low,high,close,volume,money
    price = np.insert(price[:, [0, 1, 2, 4, 5]], 1, price[:, 3], axis=1)
    df = pd.DataFrame(price, columns=['open', 'close', 'high', 'low', 'volume', 'money'])
    # 使用ta计算指标并取最近60天
    s_raw = get_indicator(df).values[-61:-1,:]
    # 归一化
    s = np.sign(s_raw) * np.log10(np.abs(s_raw + 1))/10
    # 预测
    action = context.model.predict(s)
    # 获取当前股数
    stock_amount = get_position(context.stock_code).closable
    # 获取资金
    money = context.portfolio.cash
    # 获取昨天的价格TODO：这里好像应该设置为今天价格，不过影响不大
    price = s_raw[-1,1]
    # 交易量
    quant = 0
    # 自定义环境中的交易函数
    if action[0] > 0:
        # 按钱数百分比买入
        # 当前的钱可以买多少手
        amount = money // (100 * price * (1 + 1.5e-3))
        # 实际买多少手
        quant = int(action[0] * amount)
    # 卖出
    elif action[0] < 0:
        # 当前手中有多少手
        amount = stock_amount / 100
        if amount != 0:
            # 实际卖出多少手
            quant = int(action[0] * amount)

    # 交易
    if quant!=0:
        order_lots(context.stock_code, quant)







__config__ = {
    "base": {
        "start_date": "2017-01-03",
        "end_date": "2020-02-07",
        "frequency": "1d",
        "matching_type": "current_bar",
        "benchmark": stock_code,
        "accounts": {
            "future": 1000000
        }
    },
    "extra": {
        "log_level": "verbose",
    },
}
