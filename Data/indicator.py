from ta.momentum import *
from ta.volatility import *
from ta.trend import *
from ta.volume import *
from ta.others import *
import pandas as pd
import numpy as np

# stock_list = ['000938_XSHE', '601318_XSHG', '601336_XSHG', '601601_XSHG', '601628_XSHG']
stock_list = ['601318_XSHG']
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
            raise Exception("不支持的参数个数")
        raw = pd.concat([raw, temp], axis=1)
    return raw


for stock in stock_list:
    mode = ['train', 'test']
    for m in mode:
        raw = pd.read_csv('./' + m + '/' + stock + '_day.csv')
        print("stock:" + stock)
        raw = get_indicator(raw)
        print("----fillna----")
        raw.index = list(raw['Unnamed: 0'])
        raw.pop('Unnamed: 0')
        raw.fillna(method='ffill', inplace=True)
        raw.fillna(raw.mean(), inplace=True)
        # # 做diff处理
        # data = np.array(raw)
        # data = np.diff(data,axis=0)
        # # indicator = np.array(raw)[1:, 6:]
        # index = list(raw.index)[1:]
        # # data = np.concatenate((ochlvm, indicator), axis=-1)
        # data = pd.DataFrame(data, index=index, columns=raw.columns)
        print(raw.values.shape)
        raw.to_csv('./' + m + '/' + stock + '_with_indicator.csv')
