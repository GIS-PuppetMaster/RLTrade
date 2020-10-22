from collections import OrderedDict

from jqdatasdk import *
import jqdatasdk as jq
from datetime import datetime
import datetime as dt
import pandas as pd
import numpy as np
import pickle as pk
import json
import traceback
import os
from copy import deepcopy


def get_data(stock_codes, start_date, end_date, frequency, skip_exists=False, skip_code=None):
    for stock in stock_codes:
        if skip_exists and os.path.exists(f'./raw/{stock.replace(".", "_")}.csv'):
            continue
        if skip_code is not None:
            stock_num = int(stock.split('.')[0])
            skip_stock_num = int(skip_code.split('.')[0])
            if stock_num <= skip_stock_num:
                continue
        print(stock)
        data = get_price(stock.replace("_", "."), start_date=start_date, end_date=end_date,
                         frequency=frequency, skip_paused=True, fq='pre')
        data.to_csv('./raw/' + stock.replace(".", "_") + '.csv')


def get_money_flow_(stock_codes, start_date, end_date, skip_exists=False, skip_code=None):
    stock_list = deepcopy(stock_codes)
    for stock in stock_codes:
        if skip_exists and os.path.exists(f'./raw/{stock.replace(".", "_")}_moneyflow.csv'):
            continue
        if skip_code is not None:
            stock_num = int(stock.split('.')[0])
            skip_stock_num = int(skip_code.split('.')[0])
            if stock_num <= skip_stock_num:
                continue
        print(stock)
        try:
            data = get_money_flow(stock.replace("_", "."), start_date=start_date, end_date=end_date)
            data.to_csv('./raw/' + stock.replace(".", "_") + '_moneyflow.csv')
        except Exception as e:
            traceback.print_exc()
            stock_list.remove(stock)
    return stock_list


def get_fundamentals_df(mode):
    import dill
    import math
    with open(f'./{mode}/TradeEnvData.dill', 'rb') as f:
        stock_codes_, time_series, global_date_intersection = dill.load(f)
    format_stocks = list(map(lambda x: x.replace('_', '.'), stock_codes_))
    q = query(valuation).filter(valuation.code.in_(format_stocks))
    df = None
    window_size = 10000 // len(stock_codes_)
    time_iter_times = math.ceil(len(global_date_intersection) / window_size)
    for i in range(1, time_iter_times + 1):
        last = len(global_date_intersection) - (i - 1) * window_size
        if last > window_size:
            last = window_size
            data = get_fundamentals_continuously(q, end_date=global_date_intersection[i * window_size - 1], count=last, panel=False)
        else:
            data = get_fundamentals_continuously(q, end_date=global_date_intersection[-1], count=last, panel=False)
        if df is None:
            df = data
        else:
            df = pd.concat([df, data], axis=0)
    for stock in format_stocks:
        stock_df = df[df.code == stock]
        stock_df.fillna(method='ffill', inplace=True)
        stock_df.fillna(method='bfill', inplace=True)
        stock_ = stock.replace('.', '_')
        stock_df = stock_df.reset_index(drop=True).drop(['id', 'code.1', 'day.1'], axis=1)
        stock_df.to_csv(f'./{mode}/{stock_}_fundamental.csv', index=False)
        print(f'saved {stock}')

def clean(mode):
    file_list_ = os.listdir(f'./{mode}/')
    file_list = []
    for file in file_list_:
        if 'fundamental' in file:
            file_list.append(file)
    for file in file_list:
        try:
            df = pd.read_csv(f'./{mode}/{file}').drop('code', axis=1)
            df.to_csv(f'./{mode}/{file}', index=False)
        except:
            pass


if __name__ == '__main__':
    try:
        jq.logout()
    except:
        pass
    jq.auth('15143292011', 'qwer12345')
    # get_fundamentals_df('train')
    # get_fundamentals_df('test')
    clean('train')
    clean('test')
    # with open('./000300_XSHG_list.txt', 'r') as f:
    #     stock_list = f.read()
    # stock_list = stock_list.replace("[", "").replace("]", "").replace("\'", "").replace("\"", "").replace(" ", "").split(",")
    # # stocks = get_index_stocks('000300.XSHG')
    # # stock_list = deepcopy(stocks)
    # # start_date_limit = datetime(2010, 1, 4, 0, 0, 0)
    # # start_date = datetime(2007, 1, 4, 0, 0, 0)
    # # for stock in stocks:
    # #     res = get_price(stock, start_date, start_date)
    # #     if res.index[0].to_pydatetime() > start_date_limit:
    # #         stock_list.remove(stock.replace('_','.'))
    # # with open('./000300_XSHG_list.pkl', 'wb') as f:
    # #     pk.dump(stock_list, f)
    # # with open('./000300_XSHG_list.txt', 'w') as f:
    # #     json.dump(stock_list, f)
    # print('get moneyflow')
    # stock_list = get_money_flow_(stock_list, datetime(2007, 1, 4, 0, 0, 0), datetime(2020, 8, 22, 0, 0, 0), skip_exists=False)
    # print('get data')
    # # stock_list = list(set(stock_list).difference(set(old_stocks)))
    # get_data(stock_list, datetime(2007, 1, 4, 9, 31, 0), datetime(2020, 8, 22, 15, 0, 0), '1d', skip_exists=True)
