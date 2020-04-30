from jqdatasdk import *
from datetime import datetime
import pandas as pd


def get_data(stock_codes, start_date, end_date, frequency):
    try:
        logout()
    except:
        pass
    for stock in stock_codes:
        stock = stock.replace(".", "_")
        auth('13079761737', 'ZKX741481546zkx')
        data = get_price(stock.replace("_", "."), start_date=start_date, end_date=end_date,
                         frequency=frequency, skip_paused=True, fq='pre')
        data.to_csv('./raw/' + stock + '.csv')


if __name__ == '__main__':
    get_data(['000938.XSHE'], datetime.datetime(2007, 1, 4, 9, 31, 0), datetime.datetime(2020,4,22,15,0,0), '1m')
