from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.techindicators import TechIndicators
import pandas as pd
from matplotlib import pyplot as plt

'''
ti = TechIndicators(key='SAFWRBY6D3BRFLUU', output_format='pandas', indexing_type='date')

data, meta_data = ti.get_sma(symbol='AAPL', interval='1min', time_period=20)
data.to_csv('csv/FB1min1000sma20.csv')

data, meta_data = ti.get_sma(symbol='AAPL', interval='1min', time_period=80)
data.to_csv('csv/FB1min1000sma80.csv')

data, meta_data = ti.get_stoch(symbol='AAPL', interval='1min')
data.to_csv('csv/FB1min1000stoch.csv')

ts = TimeSeries(key='SAFWRBY6D3BRFLUU',output_format='pandas', indexing_type='date')

data, meta_data = ts.get_intraday(symbol='AAPL',interval='1min', outputsize='full')
data.to_csv('csv/FB1min1000.csv')
'''


price = pd.read_csv('csv/FB1min1000.csv')
sma20 = pd.read_csv('csv/FB1min1000sma20.csv')
sma80 = pd.read_csv('csv/FB1min1000sma80.csv')
stoch = pd.read_csv('csv/FB1min1000stoch.csv')
price = price[100:275]
sma20 = sma20[81:256]
sma80 = sma80[21:196]
stoch = stoch[92:267]

price = price.reset_index()
price = price['4. close']
sma20 = sma20.reset_index()
sma20 = sma20['SMA']
sma80 = sma80.reset_index()
sma80 = sma80['SMA']
#price = price[200:1000] bear
#sma20 = sma20[181:981]
#sma80 = sma80[121:921]
#stoch = stoch[192:992]
#plt.plot(sma20['SMA'],'b')
#plt.plot(sma80['SMA'], 'g')
plt.plot(price,'r')
plt.plot(sma20,'g')
plt.plot(sma80,'b')
plt.show()
print(price)
print(sma20)
print(sma80)
print(stoch)