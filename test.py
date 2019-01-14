import requests
import io
import pandas as pd
import backtest as twp
import numpy as np
from matplotlib import pyplot as plt
import random, timeit
from sklearn import preprocessing

from keras.models import load_model

from functions import *

gamma = 0.95  # discount factor, higher means value future reward more
epsilon = 0.1


data,data_prev, sma20, sma80 = getData()
model = getModel(1)

signal = pd.Series(index=np.arange(len(data)))
signal.fillna(value=0, inplace=True)


startTime = timeit.default_timer()

state,pdata = initializeState(data, sma20, sma80)
endState=0
timeStep=1
inventory = []
profit = 0
while not endState:
    qValues = model.predict(state, batch_size=1)

    #exploitation vs exploration
    if (random.random() < epsilon):
        action = np.random.randint(0, 3)
    else:  # choose best action from Q(s,a) values
        action = (np.argmax(qValues))

    nextState, timeStep, signal, endState, profit = trade(action,pdata,signal,timeStep,inventory,data,profit)

    state = nextState

while len(inventory) > 0:
    profit += data.iloc[-1] - inventory.pop(0) #unsure if should be calculated this way??

'''long = 0
short = 0
for i in range(signal.shape[0]):
    if signal.iloc[i]<0:
        short+=1
    elif signal.iloc[i]>0:
        long+=1
print(long)
print(short)'''

bt = twp.Backtest(data, signal, signalType='shares')
endReward = bt.pnl.iloc[-1]
plt.figure(figsize=(20, 10))

print("profit is ",profit)


elapsed = np.round(timeit.default_timer() - startTime, decimals=2)
print("Completed in %f" % (elapsed,))

bt = twp.Backtest(data, signal, signalType='shares')
bt.data['delta'] = bt.data['shares'].diff().fillna(0)

print(bt.data)

plt.figure(figsize=(10, 10))
plt.subplot(2, 1, 1)
plt.title("trades")
plt.xlabel("timestamp")
bt.plotTrades()
plt.subplot(2, 1, 2)
plt.title("PnL")
plt.xlabel("timestamp")
bt.pnl.plot(style='-')
plt.tight_layout()
plt.savefig('plots/bull_test/summary_test' + '.png', bbox_inches='tight', pad_inches=1, dpi=72)
plt.show()
