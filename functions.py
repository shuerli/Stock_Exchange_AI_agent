import pandas as pd
import backtest as twp
import numpy as np
from sklearn import preprocessing

from keras.models import Sequential
from keras.layers.core import Dense, Dropout
from keras.layers.recurrent import LSTM
from keras.models import load_model

from matplotlib import pyplot as plt
# pnl explained : https://www.investopedia.com/ask/answers/how-do-you-calculate-percentage-gain-or-loss-investment/
# backtesting: a strategy to analyze how accurate a model did in performing trading with historycal data


def getModel(load):

    #number of inputs
    num_inputs = 6

    if load:
        model = load_model('model/episode960.h5')
    else:
        model = Sequential()
        model.add(LSTM(64, input_shape=(1, num_inputs), return_sequences=True, stateful=False))
        model.add(Dropout(0.2))
        model.add(LSTM(64, return_sequences=False, stateful=False))
        model.add(Dropout(0.2))
        model.add(Dense(3, activation="linear"))
        model.compile(optimizer='adam', loss='mse')
    return model

# read data into pandas dataframe from csv file
def getData():
    price = pd.read_csv('csv/FB1min1000.csv')
    # price = price.tail(100).reset_index()
    price = price[100:275]
    price = price.reset_index()
    price = price['4. close']

    price2 = pd.read_csv('csv/FB1min1000.csv')
    price2 = price2[99:274]
    price2 = price2.reset_index()
    price2 = price2['4. close']

    sma20 = pd.read_csv('csv/FB1min1000sma20.csv')
    # sma20 = sma20.tail(100).reset_index()
    sma20 = sma20[81:256]
    sma20 = sma20.reset_index()
    sma20 = sma20['SMA']
    sma80 = pd.read_csv('csv/FB1min1000sma80.csv')
    # sma80 = sma80.tail(100).reset_index()
    sma80 = sma80[21:196]
    sma80 = sma80.reset_index()
    sma80 = sma80['SMA']

    stoch = pd.read_csv('csv/FB1min1000stoch.csv')
    stoch = stoch[92:267]
    stoch = stoch.reset_index()
    slowD = stoch['SlowD']
    slowK = stoch['SlowK']

    return price, price2, sma20, sma80, slowD, slowK

# initialize first state
def initializeState(data, data_prev, sma20, sma80,slowD,slowK):

    # stack all data into a table
    pdata = np.column_stack((data, data_prev, sma20, sma80,slowD,slowK))
    pdata = np.nan_to_num(pdata)

    # pre-process data using standard scalar
    scaler = preprocessing.StandardScaler() #unit standard derivation and 0 mean
    pdata = scaler.fit_transform(pdata)

    # expand dimension to fit into the neural net input
    pdata = np.expand_dims(pdata, axis=1)

    # initial state is 1st row of the table
    initialState = pdata[0:1, :, :]

    return initialState, pdata

# reward function
def getReward(timeStep, signal, price,state, endState):

    if not endState:
        # net earning from previous action
        net = (price[timeStep] - price[timeStep - 1]) * signal[timeStep - 1]
    else:
        bt = twp.Backtest(price, signal, signalType='shares')
        net = bt.pnl.iloc[-1]
    rewards = 0


    #intuition reward
    if net > 0:
        rewards += net
    elif net < 0:
        rewards += net/2
    else:
        rewards -= 1 #don't encourage hold


    if not endState:
        #sma reward
        sma20 = state[0,0,2]
        sma80 = state[0,0,3]
        sma_net = sma20 - sma80

        if sma_net < 0 : # short sma < long sma, down trend
            if signal[timeStep - 1] < 0:
                rewards += abs(net)
        elif sma_net > 0: #short sma > long sma, up trend
            if signal[timeStep - 1] > 0:
                rewards += abs(net)





    return rewards

def trade(action, pdata, signal, timeStep, inventory, data, totalProfit):

    profit = totalProfit

    if action == 1:  # buy 1
        signal.loc[timeStep] = 10
        inventory.append(data[timeStep])
    elif action == 2:  # sell 1
        signal.loc[timeStep] = -10

        # if inventory is not empty, sell
        if len(inventory) > 0:
            profit += data[timeStep] - inventory.pop(0)
    else:
        signal.loc[timeStep] = 0

    # increase timeStep
    timeStep += 1


    # determine if it's the last state
    if timeStep  == pdata.shape[0]:
        endState = 1
        state = pdata[timeStep -1: timeStep, :, :]  # don't want go out of bound
    else:
        endState = 0
        # this is next state
        state = pdata[timeStep: timeStep + 1, :, :]  # preserves dimension for lstm input

    reward = getReward(timeStep,signal,data,state, endState)

    return state, timeStep, signal, endState, profit, reward





# test agent without random actions
def test_agent(model, data,data_prev, sma20, sma80,slowD,slowK):

    signal = pd.Series(index=np.arange(len(data)))
    signal.fillna(value=0, inplace=True)
    signal.loc[0] = 1
    state, pdata = initializeState(data, data_prev, sma20, sma80,slowD,slowK)
    endState = 0
    timeStep = 1
    realProfit = 0
    realInventory = []
    while not endState:
        Q = model.predict(state, batch_size=1)
        action = (np.argmax(Q))
        nextState, timeStep, signal, endState, realProfit,reward = trade(action, pdata, signal, timeStep, realInventory, data,
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
    print(bt.data,signal)
    plt.figure(figsize=(20, 10))
    bt.plotTrades()
    plt.suptitle(str(i))
    plt.savefig('plot/' + str(i) + '.png')
    plt.show()
    endReward = bt.pnl.iloc[-1]

    return endReward, realProfit
