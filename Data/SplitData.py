import pandas as pd
import os
from datetime import datetime

source_path = './raw'
stock_code = '601318_XSHG'
train_time = datetime(2016, 12, 30, 15, 0, 0).strftime("%Y-%m-%d")
test_time = datetime(2017, 1, 3, 9, 31, 0).strftime("%Y-%m-%d")
df = pd.read_csv(os.path.join(source_path, stock_code + '.csv'))

df.set_index(["Unnamed: 0"], inplace=True)
# source_raw.sort_index(axis=0,ascending=True)
# 截断train_time之后的
new_train = df.truncate(after=train_time)
# 截断test_time之前的
new_test = df.truncate(before=test_time)
new_train.to_csv('./train/'+stock_code+'_day.csv')
new_test.to_csv('./test/'+stock_code+'_day.csv')