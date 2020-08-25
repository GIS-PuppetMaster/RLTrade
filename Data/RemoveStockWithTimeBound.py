import pickle as pk
import os
import pandas as pd
from copy import deepcopy
from datetime import datetime
import json
path = './raw/'
time_format = '%Y-%m-%d'
start_date_limit = datetime(2010, 1, 4, 0, 0, 0)

with open('./000300_XSHG_list.pkl', 'rb') as f:
    stock_codes = pk.load(f)
stock_list = deepcopy(stock_codes)
for stock_code in stock_codes:
    filename = stock_code.replace('.', '_') + '.csv'
    df = pd.read_csv(path + filename, index_col='Unnamed: 0')
    if datetime.strptime(df.index[0],time_format) > start_date_limit:
        stock_list.remove(stock_code)
with open('./000300_XSHG_list.pkl', 'wb') as f:
    pk.dump(stock_list, f)
with open('./000300_XSHG_list.txt', 'w') as f:
    json.dump(stock_list, f)
