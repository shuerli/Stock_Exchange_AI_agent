import pandas as pd
import backtest as twp

from functions import getData
import numpy as np
# get price & techinical indicator data as pandas dataframe
data, data_prev, sma20, sma80,slowD,slowK = getData()
signal = pd.Series(index=np.arange(len(data)))

for i in range(signal.shape[0]):
    signal.loc[i] = 10
#signal.loc[1] = 0
signal.loc[0] = 1
print(signal)
bt = twp.Backtest(pd.Series(data=[x for x in data], index=signal.index.values), signal, signalType='shares')
endReward = bt.pnl.iloc[-1]
print(bt.data)
print(signal.shape[0])
print(endReward)