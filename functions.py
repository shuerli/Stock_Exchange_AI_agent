import pandas as pd
import backtest as twp
import numpy as np
from matplotlib import pyplot as plt
import random, timeit
from sklearn import preprocessing

from keras.models import Sequential
from keras.layers.core import Dense, Dropout
from keras.layers.recurrent import LSTM
from keras.models import load_model


# pnl explained : https://www.investopedia.com/ask/answers/how-do-you-calculate-percentage-gain-or-loss-investment/
# backtesting: a strategy to analyze how accurate a model did in performing trading with historycal data
def getData():
    price = pd.read_csv('csv/FB1min1000.csv')
    # price = price.tail(100).reset_index()
    price = price[900:1400]
    price = price.reset_index()
    price = price['4. close']

    price2 = pd.read_csv('csv/FB1min1000.csv')
    price2 = price2[899:1399]
    price2 = price2.reset_index()
    price2 = price2['4. close']

    sma20 = pd.read_csv('csv/FB1min1000sma20.csv')
    # sma20 = sma20.tail(100).reset_index()
    sma20 = sma20[900:1400]
    sma20 = sma20.reset_index()
    sma20 = sma20['SMA']
    sma80 = pd.read_csv('csv/FB1min1000sma80.csv')
    # sma80 = sma80.tail(100).reset_index()
    sma80 = sma80[900:1400]
    sma80 = sma80.reset_index()
    sma80 = sma80['SMA']
    return price, price2, sma20, sma80


def initializeState(data, data_prev, sma20, sma80):
    # Preprocessing Data
    pdata = np.column_stack((data, data_prev, sma20, sma80))
    pdata = np.nan_to_num(pdata)
    scaler = preprocessing.StandardScaler()
    pdata = scaler.fit_transform(pdata)
    pdata = np.expand_dims(scaler.fit_transform(pdata), axis=1)  # add dimension for lstm input
    initialState = pdata[1:2, :, :]

    return initialState, pdata


def trade(action, pdata, signal, timeStep, inventory, data, totalProfit):
    profit = totalProfit
    timeStep += 1
    state = pdata[timeStep: timeStep + 1, :, :]  # preserves dimension for lstm input
    if timeStep + 1 == pdata.shape[0]:
        endState = 1
    else:
        endState = 0

    if action == 1:  # buy 1
        signal.loc[timeStep - 1] = 1
        inventory.append(data[timeStep - 1])
    elif action == 2:  # and len(inventory) > 0: #sell 1
        signal.loc[timeStep - 1] = -1
        if len(inventory) > 0:
            profit += data[timeStep - 1] - inventory.pop(0)
    else:
        signal.loc[timeStep - 1] = 0

    return state, timeStep, signal, endState, profit


def getReward(timeStep, signal, endState, price):
    net = (price[timeStep] - price[timeStep - 1]) * signal[timeStep - 1]
    rewards = 0
    if not endState:

        if net > 0:
            rewards = 1
        elif net < 0:
            rewards = -1

    else:
        bt = twp.Backtest(price, signal, signalType='shares')
        rewards = bt.pnl.iloc[-1]

    return rewards


def getModel(load):
    num_inputs = 4


    if load:
        model = load_model('model/episode400.h5')
    else:
        model = Sequential()
        model.add(LSTM(80, input_shape=(1, num_inputs), return_sequences=True, stateful=False))
        model.add(Dropout(0.2))
        model.add(LSTM(80, return_sequences=False, stateful=False))
        model.add(Dropout(0.2))
        model.add(Dense(3, activation="linear"))
        model.compile(optimizer='adam', loss='mse')
    return model

# test agent without random actions
def test(model, data,data_prev, sma20, sma80):
    signal = pd.Series(index=np.arange(len(data)))
    signal.fillna(value=0, inplace=True)

    state, pdata = initializeState(data, data_prev, sma20, sma80)
    endState = 0
    timeStep = 1
    realProfit = 0
    realInventory = []
    while not endState:
        Q = model.predict(state, batch_size=1)
        action = (np.argmax(Q))
        # print(Q,'***',action)
        nextState, timeStep, signal, endState, realProfit = trade(action, pdata, signal, timeStep, realInventory, data,
                                                                  realProfit)

        state = nextState
    while len(realInventory) > 0:
        realProfit += data.iloc[-1] - realInventory.pop(0)

    long = 0
    short = 0
    hold = 0
    for i in range(signal.shape[0]):
        if signal.iloc[i] < 0:
            short += 1
        elif signal.iloc[i] > 0:
            long += 1
        else:
            hold+=1
    print('r-Buy: ', long, ', r-Sell: ', short, 'r-hold: ',hold)

    bt = twp.Backtest(data, signal, signalType='shares')
    endReward = bt.pnl.iloc[-1]

    return endReward, realProfit
