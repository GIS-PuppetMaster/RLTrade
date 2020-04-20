import pandas as pd
import os
from datetime import datetime

source_path = 'E:\PycharmProjects\Data'
stock_code = '000938_XSHE'
train_time = datetime(2016, 12, 30, 15, 0, 0).strftime("%Y-%m-%d %H:%M:%S")
test_time = datetime(2017, 1, 3, 9, 31, 0).strftime("%Y-%m-%d %H:%M:%S")

source_train = pd.read_csv(os.path.join(source_path, 'train', stock_code+'.csv'))
source_test = pd.read_csv(os.path.join(source_path, 'test', stock_code+'.csv'))
source_train.set_index(["Unnamed: 0"], inplace=True)
source_test.set_index(["Unnamed: 0"], inplace=True)
source_raw = pd.concat([source_train, source_test], axis=0)
# source_raw.sort_index(axis=0,ascending=True)
new_train = source_raw.truncate(after=train_time)
new_test = source_raw.truncate(before=test_time)
new_train.to_csv('./train/'+stock_code+'.csv')
new_test.to_csv('./test/'+stock_code+'.csv')
# day_data
train_time = datetime(2016, 12, 30, 0, 0, 0).strftime("%Y-%m-%d")
test_time = datetime(2017, 1, 3, 0, 0, 0).strftime("%Y-%m-%d")

source_train = pd.read_csv(os.path.join(source_path, 'train', stock_code+'_day.csv'))
source_test = pd.read_csv(os.path.join(source_path, 'test', stock_code+'_day.csv'))
source_train.set_index(["Unnamed: 0"], inplace=True)
source_test.set_index(["Unnamed: 0"], inplace=True)
source_raw = pd.concat([source_train, source_test], axis=0)
# source_raw.sort_index(axis=0,ascending=True)
new_train = source_raw.truncate(after=train_time)
new_test = source_raw.truncate(before=test_time)
new_train.to_csv('./train/'+stock_code+'_day.csv')
new_test.to_csv('./test/'+stock_code+'_day.csv')