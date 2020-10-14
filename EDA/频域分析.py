import pywt
import numpy as np
import pandas as pd
import dill
import matplotlib.pyplot as plt
import seaborn as sn

stock_data_path = '../Data/train/'
stock_code = '000001_XSHE'
raw = pd.read_csv(stock_data_path + stock_code + '_with_indicator.csv', index_col=False)
raw_moneyflow = pd.read_csv(stock_data_path + stock_code + '_moneyflow.csv', index_col=False)[
    ['date', 'change_pct', 'net_pct_main', 'net_pct_xl', 'net_pct_l', 'net_pct_m', 'net_pct_s']].apply(
    lambda x: x / 100 if isinstance(x[1], np.float64) else x)
raw = pd.merge(raw, raw_moneyflow, left_on='Unnamed: 0', right_on='date', sort=False, copy=False).drop(
    'date', 1).rename(columns={'Unnamed: 0': 'date'})
raw.fillna(method='ffill', inplace=True)
raw.set_index('date', inplace=True)
data = raw.values
wave = pywt.wavedec(data[:, 0], 'haar')
for i in range(len(wave)):
    res = list(map(lambda x: np.expand_dims(x, axis=0), list(wave[i])))
    res = np.concatenate(res, axis=0)
    res = np.concatenate([np.arange(res.shape[0]).reshape(-1, 1), res.reshape(-1, 1)], axis=-1)
    sn.lineplot(data=pd.DataFrame(res, columns=['x', 'y']), x='x', y='y')
    plt.show()
res = np.concatenate([np.arange(data.shape[0]).reshape(-1, 1), data[:, 0].reshape(-1, 1)], axis=-1)
sn.lineplot(data=pd.DataFrame(res, columns=['x', 'y']), x='x', y='y')
plt.show()
