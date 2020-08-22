from rqalpha import run_file
import os
from Test import find_model
from ta.momentum import *
from ta.volatility import *
from ta.trend import *
from ta.volume import *
from ta.others import *
from Util.Util import post_processor

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
skip_suspended = True
id = "aq8kk9kk"
type = "best"
path = './RQStrategyTest.py'

config = {
    "base": {
        "start_date": "2017-01-03",
        "end_date": "2020-07-07",
        "frequency": "1d",
        "accounts": {
            "stock": 1e5
        },
    },
    "extra": {
        "log_level": "verbose",
    },
    "mod": {
        "sys_analyser": {
            "enabled": True,
            "plot": True
        },
        "sys_accounts": {
            "enabled": True,
            # 开启/关闭 股票 T+1， 默认开启
            "stock_t1": True,
            # 分红再投资
            "dividend_reinvestment": False,
            # 当持仓股票退市时，按照退市价格返还现金
            "cash_return_by_stock_delisted": True,
            # 股票下单因资金不足被拒时改为使用全部剩余资金下单
            "auto_switch_order_value": True,
            # 检查股票可平仓位是否充足
            "validate_stock_position": True,
            # 检查期货可平仓位是否充足
            "validate_future_position": True,
        },
        "sys_simulation": {
            "enabled": True,
            "slippage": 0.,
            "matching_type": "next_bar",
        }
    }
}
if __name__ == '__main__':
    folder_name, _, max_file_name = find_model(id, type)
    import yaml
    with open(os.path.join('./wandb', folder_name, 'config.yaml'), 'r') as f:
        conf = f.read()
    conf = yaml.load(conf)
    conf['agent_config'] = conf['agent_config']['value']
    conf['train_env_config'] = conf['train_env_config']['value']
    conf['eval_env_config'] = conf['eval_env_config']['value']
    if conf['train_env_config']['post_processor'] == 'post_processor':
        conf['train_env_config']['post_processor'] = post_processor
    else:
        raise Exception("train_env_config:post_processor为不支持的类型{}".format(conf['train_env_config']['post_processor']))
    if conf['eval_env_config']['post_processor'] == 'post_processor':
        conf['eval_env_config']['post_processor'] = post_processor
    else:
        raise Exception("eval_env_config:post_processor为不支持的类型{}".format(conf['eval_env_config']['post_processor']))
    globals().update(conf)
    globals().update(conf['agent_config'])
    for stock_code in conf['train_env_config']['stock_codes']:
        stock_code = stock_code.replace("_", ".")
        config['base']['benchmark'] = stock_code
        plot_save_path = os.path.join(os.getcwd(), 'TestResult', folder_name, max_file_name,
                                      stock_code + ".png")
        file_save_path = os.path.join(os.getcwd(), 'TestResult', folder_name, max_file_name,
                                      stock_code + ".pkl")
        if not os.path.exists(os.path.join(os.getcwd(), 'TestResult', folder_name, max_file_name)):
            os.makedirs(os.path.join(os.getcwd(), 'TestResult', folder_name, max_file_name))
        config['base']['stock_code'] = stock_code
        config['mod']['sys_analyser']['plot_save_file'] = plot_save_path
        config['mod']['sys_analyser']['output_file'] = file_save_path
        run_file(path, config)
