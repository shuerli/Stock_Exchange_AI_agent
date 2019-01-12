from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.techindicators import TechIndicators
import pandas as pd
from matplotlib import pyplot as plt


ti = TechIndicators(key='SAFWRBY6D3BRFLUU', output_format='pandas', indexing_type='date')

data, meta_data = ti.get_sma(symbol='FB', interval='1min', time_period=20)
data.to_csv('csv/bull_test/FB1min1000sma20.csv')

data, meta_data = ti.get_sma(symbol='FB', interval='1min', time_period=80)
data.to_csv('csv/bull_test/FB1min1000sma80.csv')



ts = TimeSeries(key='SAFWRBY6D3BRFLUU',output_format='pandas', indexing_type='date')

data, meta_data = ts.get_intraday(symbol='FB',interval='1min', outputsize='full')
data.to_csv('csv/bull_test/FB1min1000.csv')



price = pd.read_csv('csv/bull_test/FB1min1000.csv')
sma20 = pd.read_csv('csv/bull_test/FB1min1000sma20.csv')
sma80 = pd.read_csv('csv/bull_test/FB1min1000sma80.csv')

price = price[900:1400]

#plt.plot(sma20['SMA'],'b')
#plt.plot(sma80['SMA'], 'g')
plt.plot(price['4. close'],'r')
plt.show()
print(price)
print(sma20)
print(sma80)