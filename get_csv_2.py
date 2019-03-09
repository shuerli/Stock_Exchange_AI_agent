'''
import pandas as pd
from matplotlib import pyplot as plt
from functions import getData
import numpy as np
from sklearn import preprocessing
'''
#stock = 'MMM'
#time = '1min'




'''
price = pd.read_csv('csv/price.csv')
sma20 = pd.read_csv('csv/sma20.csv')
sma80 = pd.read_csv('csv/sma80.csv')
stoch = pd.read_csv('csv/stoch.csv')
rsi = pd.read_csv('csv/rsi.csv')
dji = pd.read_csv('csv/dji.csv')
'''
#sma20 = price - 19
#sma80 = price - 79
#stoch = price - 8
#rsi = price - 20

#print(price)
#print(sma20)
#print(sma80)
#print(stoch)
#rint(rsi)
#print(dji)

'''
#price = price[1200:1800]
price = price[900:1200]
#price = price.tail(1000)
print(price.iloc[0],'\n',price.iloc[-1],'\n')
print('*****',price.shape[0])
price = price.reset_index()
price = price['4. close']

sma20 = sma20[881:1181]
#sma20 = sma20.tail(1000)
print(sma20.iloc[0],'\n',sma20.iloc[-1],'\n')
print('*****',sma20.shape[0])
print(sma20)
#print(sma20.to_string())
sma20 = sma20.reset_index()
sma20 = sma20['SMA']

sma80 = sma80[820:1119]
#sma80 = sma80.tail(1000)
print(sma80.iloc[0],'\n',sma80.iloc[-1],'\n')
print('*****',sma80.shape[0])
print(sma80)
#print(sma80.to_string())
sma80 = sma80.reset_index()
sma80 = sma80['SMA']

dji = dji[899:1207]
#dji = dji.tail(1000)
print(dji.iloc[0],'\n',dji.iloc[-1],'\n')
#print(dji.to_string())

stoch = stoch[892:1192]
#stoch = stoch.tail(1000)
print(stoch.iloc[0],'\n',stoch.iloc[-1],'\n')
#print(stoch.to_string())

rsi = rsi[879:1178]
#rsi = rsi.tail(1000)
print(rsi.iloc[0],'\n',rsi.iloc[-1],'\n')

dji = dji.reset_index()
dji = dji['4. close']
#price = np.expand_dims(price,axis=1)
dji = np.expand_dims(dji,axis=1)
scaler = preprocessing.StandardScaler() #unit standard derivation and 0 mean
dji = scaler.fit_transform(dji)
#price = scaler.fit_transform(price)
'''

'''
# get price & techinical indicator data as pandas dataframe
pdata,data = getData(2)

print(data)
plt.plot(data,'r')
#plt.plot(sma20,'g')
#plt.plot(sma80,'b')
#plt.plot(dji)
plt.savefig('csv/stock.png')
plt.show()

'''
