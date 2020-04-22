from jqdatasdk import *
from datetime import datetime
import pandas as pd
def get_data(code,start_date,end_date,frequency):
    try:
        logout()
    except:
        pass
    auth('13079761737', 'ZKX741481546zkx')
    data = get_price(code, start_date=start_date, end_date=end_date,
                     frequency=frequency, skip_paused=True, fq='pre')
    data.to_csv('./raw/' + code.replace(".", "_")+'.csv')
