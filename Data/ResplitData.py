import pandas as pd
import os
import pickle as pk
from datetime import datetime

source_path = 'E:\PycharmProjects\Data'


def split_data(stock_code, train_time, test_time, postfix):
    source_train = pd.read_csv(os.path.join(source_path, 'train', stock_code + postfix))
    source_test = pd.read_csv(os.path.join(source_path, 'test', stock_code + postfix))
    source_train.set_index(["Unnamed: 0"], inplace=True)
    source_test.set_index(["Unnamed: 0"], inplace=True)
    source_raw = pd.concat([source_train, source_test], axis=0)
    # source_raw.sort_index(axis=0,ascending=True)
    new_train = source_raw.truncate(after=train_time)
    new_test = source_raw.truncate(before=test_time)
    new_train.to_csv('./train/' + stock_code + postfix)
    new_test.to_csv('./test/' + stock_code + postfix)


if __name__ == '__main__':
    with open('./000300_XSHG_list.pkl', 'rb') as f:
        stock_list = pk.load(f)
    for stock in stock_list:
        train_time = datetime(2016, 12, 30, 15, 0, 0).strftime("%Y-%m-%d %H:%M:%S")
        test_time = datetime(2017, 1, 3, 9, 31, 0).strftime("%Y-%m-%d %H:%M:%S")
        split_data(train_time, test_time, '.csv')
        # day_data
        train_time = datetime(2016, 12, 30, 0, 0, 0).strftime("%Y-%m-%d")
        test_time = datetime(2017, 1, 3, 0, 0, 0).strftime("%Y-%m-%d")
        split_data(train_time, test_time, '_day.csv')
        # split_data(stock.replace(".", "_"), train_time, test_time, '_moneyflow.csv')
