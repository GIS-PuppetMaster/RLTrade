from jqdatasdk import *
from datetime import datetime
import pandas as pd
try:
    logout()
except:
    pass
auth('13079761737', 'ZKX741481546zkx')
code = '601318.XSHG'
data = get_price(code, start_date=datetime(2007, 3, 1, 0, 0, 0), end_date=datetime(2020, 4, 20, 0, 0, 0),
                 frequency='1d', skip_paused=False, fq='pre', fill_paused=True)
data.to_csv('./raw/' + code.replace(".", "_")+'.csv')
