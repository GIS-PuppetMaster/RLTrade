# coding=utf-8
from stable_baselines import TRPO
from sklearn.preprocessing import StandardScaler
from RunRQ import *
from Util.Util import *




def get_indicator(raw):
    for indicator_name, fun, col_name in indicators:
        # print(indicator_name)
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
    # strategy_file_path = context.config.base.strategy_file
    # csv_path = os.path.join(os.path.dirname(strategy_file_path), "./Data/test/000938_XSHE_day.csv")
    # context.XSHE000938_df = read_csv_as_df(csv_path)
    import os
    strategy_file_path = context.config.base.strategy_file

    folder_name, model_path, _ = find_model(id, type, os.path.dirname(strategy_file_path))
    # 恢复配置文件
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
    logger.info('model_path:' + model_path)
    model = LoadCustomPolicyForTest(model_path)
    context.model = model
    context.stock_code = context.config.base.stock_code
    context.scaler = StandardScaler()


def before_trading(context):
    pass


def handle_bar(context, bar):
    # 获取datetime, open,high,low,close,money，不包括当天数据
    price = history_bars(context.stock_code, 120, '1d', include_now=False, skip_suspended=skip_suspended)
    # 剔除datetime
    price = pd.DataFrame(price).values[:, 1:]
    # 获取volume
    volume = history_bars(context.stock_code, 120, '1d', 'total_turnover', include_now=False,
                          skip_suspended=skip_suspended)
    # 拼接volume：open,high,low,close,volume,money
    price = np.insert(price, 4, volume, axis=1)
    # 调整顺序为open,close,high,low,volume,money
    price = np.insert(price[:, [0, 1, 2, 4, 5]], 1, price[:, 3], axis=1)
    df = pd.DataFrame(price, columns=['open', 'close', 'high', 'low', 'volume', 'money'])
    # 使用ta计算指标并取最近60天
    s_raw = get_indicator(df).values[-60:, :]
    s = context.scaler.fit_transform(s_raw)
    s = s.flatten()
    # 获取当前股数
    stock_amount = get_position(context.stock_code).closable
    # 获取资金
    money = context.portfolio.cash
    # 归一化agent状态并添加到s中
    if agent_state=='True':
        s = np.append(s, log10plus1R(np.array([money, stock_amount])) / 10)
    # 归一化
    # s = log10plus1R(s)/10
    # 预测
    # s = s[:-2].reshape([60, 26])
    action = context.model.predict(s)[0]
    logger.info(
        "环境时间: " + str(context.now) + "\nmoney: " + str(money) + "\namount: " + str(stock_amount) + "\naction: " + str(
            action))
    # 下单
    # order_percent(context.stock_code, action[0])
    # 获取昨天的价格
    price = s_raw[-1, 1]
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
    if quant != 0:
        order_lots(context.stock_code, quant)

