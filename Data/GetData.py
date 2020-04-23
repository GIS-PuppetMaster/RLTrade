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
        data.to_csv('./raw/' + stock +'.csv')
