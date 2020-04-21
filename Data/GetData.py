from jqdatasdk import *
from datetime import datetime
import pandas as pd
try:
    logout()
except:
    pass
auth('13079761737', 'ZKX741481546zkx')
code = '601318.XSHG'
data = get_price('601318.XSHG', start_date=datetime(2007, 3, 1, 0, 0, 0), end_date=(2020, 4, 21, 0, 0, 0),
                 frequency='1d', skip_paused=False, fq='pre', fill_paused=True)
data.to_csv('./raw/' + code.replace(".", "_")+'.csv')
