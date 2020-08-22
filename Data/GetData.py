from jqdatasdk import *
from datetime import datetime
import pandas as pd
import numpy as np
from copy import deepcopy

def get_data(stock_codes, start_date, end_date, frequency, start_date_limit):
    stock_list = deepcopy(stock_codes)
    for stock in stock_codes:
        stock = stock.replace(".", "_")
        data = get_price(stock.replace("_", "."), start_date=start_date, end_date=end_date,
                         frequency=frequency, skip_paused=True, fq='pre')
        if data.index[0].to_pydatetime()>start_date_limit:
            stock_list.remove(stock)
        else:
            data.to_csv('./raw/' + stock + '.csv')
    return stock_list

if __name__ == '__main__':
    get_data(['000938.XSHE'], datetime(2007, 1, 4, 9, 31, 0), datetime(2020,4,22,15,0,0), '1m')
