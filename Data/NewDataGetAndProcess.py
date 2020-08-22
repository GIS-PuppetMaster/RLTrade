from Data.GetData import *
from Data.SplitData import *
from Data.GetIndicator import *
from datetime import datetime
import jqdatasdk as jq
import pickle as pk

try:
    jq.logout()
except:
    pass
jq.auth('13074581737', 'trustno1')
stock_code = ['000938_XSHE', '002230_XSHE', '002415_XSHE', '000063_XSHE']
index = '000300.XSHG'
# stock_code = jq.get_index_stocks(index)
stock_code = get_data(stock_code, datetime(2007, 5, 1, 0, 0, 0), datetime(2020, 7, 16, 15, 0, 0), '1d', datetime(2010,1,1))
stock_code = split_data(stock_code)
with open(index.replace(".", "_") + '_list.pkl', 'wb') as f:
    pk.dump(stock_code, f)
get_and_save_indicator(stock_code)
