from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.techindicators import TechIndicators
import pandas as pd
from matplotlib import pyplot as plt

stock = 'GOOG'
time = '1min'

ti = TechIndicators(key='H7M3SZXXJ81BCUQD', output_format='pandas', indexing_type='date')
'''
data, meta_data = ti.get_sma(symbol=stock, interval=time, time_period=20)
data.to_csv('csv/sma20.csv')

data, meta_data = ti.get_sma(symbol=stock, interval=time, time_period=80)
data.to_csv('csv/sma80.csv')

data, meta_data = ti.get_stoch(symbol=stock, interval=time)
data.to_csv('csv/stoch.csv')

data, meta_data = ti.get_rsi(symbol=stock, interval=time)
data.to_csv('csv/rsi.csv')
'''
'''
ts = TimeSeries(key='SAFWRBY6D3BRFLUU',output_format='pandas', indexing_type='date')

data, meta_data = ts.get_intraday(symbol=stock,interval=time, outputsize='full')
data.to_csv('csv/price.csv')

data, meta_data = ts.get_intraday(symbol='DJI',interval=time, outputsize='full')
data.to_csv('csv/dji.csv')
'''

price = pd.read_csv('csv/price.csv')
sma20 = pd.read_csv('csv/sma20.csv')
sma80 = pd.read_csv('csv/sma80.csv')
stoch = pd.read_csv('csv/stoch.csv')
rsi = pd.read_csv('csv/rsi.csv')
dji = pd.read_csv('csv/dji.csv')

#sma20 = price - 19
#sma80 = price - 79
#stoch = price - 8
#rsi = price - 20

#print(price)
#print(sma20)
#print(sma80)
#print(stoch)
#print(rsi)
#print(dji)



price = price[400:600]
price = price.reset_index()
price = price['4. close']

sma20 = sma20[381:581]
sma20 = sma20.reset_index()
sma20 = sma20['SMA']

sma80 = sma80[321:521]
sma80 = sma80.reset_index()
sma80 = sma80['SMA']
#dji = dji.reset_index()
#dji = dji['4. close']

plt.plot(price,'r')
plt.plot(sma20,'g')
plt.plot(sma80,'b')
#plt.plot(dji)
plt.show()

