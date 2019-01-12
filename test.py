import requests
import io
import pandas as pd
import backtest as twp
import numpy as np
from matplotlib import pyplot as plt
import random, timeit
from sklearn import preprocessing

from keras.models import load_model

# pnl explained : https://www.investopedia.com/ask/answers/how-do-you-calculate-percentage-gain-or-loss-investment/
#backtesting: a strategy to analyze how accurate a model did in performing trading with historycal data
def getData():
    price = pd.read_csv('csv/bull_test/FB1min1000.csv')
    #price = price.tail(100).reset_index()
    price = price[:350]
    price = price.reset_index()
    price = price['4. close']
    sma20 = pd.read_csv('csv/bull_test/FB1min1000sma20.csv')
    #sma20 = sma20.tail(100).reset_index()
    sma20 = sma20[:350]
    sma20 = sma20.reset_index()
    sma20 = sma20['SMA']
    sma80 = pd.read_csv('csv/bull_test/FB1min1000sma80.csv')
   #sma80 = sma80.tail(100).reset_index()
    sma80 = sma80[:350]
    sma80 = sma80.reset_index()
    sma80 = sma80['SMA']
    return price, sma20,sma80

def initializeState(data, sma20, sma80):
    diff = np.diff(data)
    diff = np.insert(diff, 0, 0)

    # Preprocessing Data
    pdata = np.column_stack((data, diff, sma20, sma80))
    pdata = np.nan_to_num(pdata)
    scaler = preprocessing.StandardScaler()
    pdata = scaler.fit_transform(pdata)

    pdata = np.expand_dims(scaler.fit_transform(pdata), axis=1)  # add dimension for lstm input

    initialState = pdata[1:2, :, :]

    return initialState, pdata

def trade(action, pdata, signal, timeStep,inventory,data,totalProfit):
    profit = totalProfit
    timeStep += 1
    state = pdata[timeStep: timeStep + 1, :,:]  # preserves dimension
    if timeStep +1 == pdata.shape[0]:
        endState = 1
    else:
        endState = 0
    if action == 1: #buy
        signal.loc[timeStep-1] = 1
        inventory.append(data[timeStep-1])
    elif action == 2:# and len(inventory) > 0: #sell
        signal.loc[timeStep-1] = -1
        if len(inventory) > 0:
            profit += data[timeStep-1] - inventory.pop(0)
    else:
        signal.loc[timeStep-1] = 0

    return state, timeStep, signal, endState, profit


def getModel():
    model = load_model('model/bull_test/FINAL_model.h5')
    #model = load_model('model/bull_test/episode380.h5')
    #model = load_model('results/goog.1min.100.bull/episode400.h5')
    return model

# Main program start


gamma = 0.95  # since the reward can be several time steps away, make gamma high
epsilon = 0.1
batchSize = 10
buffer = 20
replay = []
replayIter = 0
pnl_progress = []
profit_progress = []


data, sma20, sma80 = getData()
model = getModel()

signal = pd.Series(index=np.arange(len(data)))



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
