import pandas as pd

price = pd.read_csv('csv/price.csv')
sma20 = pd.read_csv('csv/sma20.csv')
sma80 = pd.read_csv('csv/sma80.csv')
stoch = pd.read_csv('csv/stoch.csv')
rsi = pd.read_csv('csv/rsi.csv')
dji = pd.read_csv('csv/dji.csv')


price = price[['date','4. close']]
price.columns = ['date','price']
sma20 = sma20[['date', 'SMA']]
sma20.columns = ['date','sma20']
sma80 = sma80[['date', 'SMA']]
sma80.columns = ['date','sma80']
dji = dji[['date','4. close']]
dji.columns = ['date','dji']

indicator = pd.merge(sma80,sma20,on='date')
indicator = pd.merge(indicator,stoch, on='date')
indicator = pd.merge(indicator,rsi,on='date')

stock = pd.merge(price,dji,on='date')

stock['date'] = pd.to_datetime(stock.date)
indicator['date'] = pd.to_datetime(indicator.date)

final = pd.merge(stock,indicator, on='date')


print(price)
print(sma20)

print(sma80)
print(stoch)
print(rsi)
#print(dji)

print(indicator)
print(stock)
print(final)
