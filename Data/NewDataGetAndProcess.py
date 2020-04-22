from Data.GetData import *
from Data.SplitData import *
from Data.GetIndicator import *
from datetime import datetime

stock_code = '000938.XSHE'
get_data(stock_code, datetime(2007, 1, 4, 0, 0, 0), datetime(2020, 4, 22, 0, 0, 0), '30m')
split_data(stock_code)
get_and_save_indicator(stock_code)
