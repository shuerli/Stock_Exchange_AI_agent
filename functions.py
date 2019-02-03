import pandas as pd
import backtest as twp
import numpy as np
from sklearn import preprocessing

from keras.models import Sequential
from keras.layers.core import Dense, Dropout
from keras.layers.recurrent import LSTM
from keras.models import load_model
from keras.optimizers import Adam
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# pnl explained : https://www.investopedia.com/ask/answers/how-do-you-calculate-percentage-gain-or-loss-investment/
# backtesting: a strategy to analyze how accurate a model did in performing trading with historycal data


def getModel(load):

    #number of inputs
    num_inputs = 8

    if load:
        model = load_model('model/episode960.h5')
    else:
        '''
        model = Sequential()
        model.add(LSTM(64, input_shape=(1, num_inputs), return_sequences=True, stateful=False))
        model.add(Dropout(0.2))
        model.add(LSTM(64, return_sequences=False, stateful=False))
        model.add(Dropout(0.2))
        model.add(Dense(3, activation="linear"))
        model.compile(optimizer='adam', loss='mse')'''
        model = Sequential()
        model.add(Dense(units=64, input_dim=num_inputs, activation="relu"))
        model.add(Dense(units=32, activation="relu"))
        model.add(Dense(units=8, activation="relu"))
        model.add(Dense(3, activation="linear"))
        model.compile(loss="mse", optimizer=Adam(lr=0.001))
    return model

# read data into pandas dataframe from csv file
def getData():
    price = pd.read_csv('csv/price.csv')
    price = price[400:600]
    price = price.reset_index()
    price = price['4. close']

    price2 = pd.read_csv('csv/price.csv')
    price2 = price2[399:599]
    price2 = price2.reset_index()
    price2 = price2['4. close']

    sma20 = pd.read_csv('csv/sma20.csv')
    sma20 = sma20[381:581]
    sma20 = sma20.reset_index()
    sma20 = sma20['SMA']

    sma80 = pd.read_csv('csv/sma80.csv')
    sma80 = sma80[321:521]
    sma80 = sma80.reset_index()
    sma80 = sma80['SMA']

    stoch = pd.read_csv('csv/stoch.csv')
    stoch = stoch[392:592]
    stoch = stoch.reset_index()
    slowD = stoch['SlowD']
    slowK = stoch['SlowK']

    rsi = pd.read_csv('csv/rsi.csv')
    rsi = rsi[380:580]
    rsi = rsi.reset_index()
    rsi = rsi['RSI']

    dji = pd.read_csv('csv/dji.csv')
    dji = dji[400:600]
    dji = dji.reset_index()
    dji = dji['4. close']

    return price, price2, sma20, sma80, slowD, slowK, rsi, dji

# initialize first state
def initializeState(data, data_prev, sma20, sma80,slowD,slowK,rsi, dji):

    # stack all data into a table
    pdata = np.column_stack((data, data_prev, sma20, sma80,slowD,slowK, rsi, dji))
    pdata = np.nan_to_num(pdata)

    # pre-process data using standard scalar
    scaler = preprocessing.StandardScaler() #unit standard derivation and 0 mean
    pdata = scaler.fit_transform(pdata)

    # expand dimension to fit into the neural net input
    #pdata = np.expand_dims(pdata, axis=1)

    # initial state is 1st row of the table
    #initialState = pdata[0:1, :, :]
    initialState = pdata[0:1, :]

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
        rewards += net/5
    else:
        #rewards -= 1 #don't encourage hold
        rewards += 0



    # sma reward
    if not endState:
        #sma20 = state[0,0,2]
        #sma80 = state[0,0,3]
        sma20 = state[0, 2]
        sma80 = state[0,3]
        sma_net = sma20 - sma80

        if sma_net < 0 : # short sma < long sma, down trend
            if signal[timeStep - 1] < 0:
                rewards += abs(sma_net)
        elif sma_net > 0: #short sma > long sma, up trend
            if signal[timeStep - 1] > 0:
                rewards += abs(sma_net)





    return rewards

def trade(action, pdata, signal, timeStep, inventory, data, totalProfit):

    profit = totalProfit

    if action == 1:  # buy 1
        signal.loc[timeStep] = 10
        inventory.append(data[timeStep])
    elif action == 2:  # sell 1
        signal.loc[timeStep] = -10

        # if inventory is not empty, sell all
        if len(inventory) > 0:
            while len(inventory) > 0:
                profit += data[timeStep] - inventory.pop(0)
    else:
        signal.loc[timeStep] = 0

    # increase timeStep
    timeStep += 1


    # determine if it's the last state
    if timeStep  == pdata.shape[0]:
        endState = 1
        #state = pdata[timeStep -1: timeStep, :, :]  # don't want go out of bound
        state = pdata[timeStep - 1: timeStep, :]
    else:
        endState = 0
        # this is next state
        #state = pdata[timeStep: timeStep + 1, :, :]  # preserves dimension for lstm input
        state = pdata[timeStep: timeStep + 1, :]

    reward = getReward(timeStep,signal,data,state, endState)

    return state, timeStep, signal, endState, profit, reward





# test agent without random actions
def test_agent(model, data,data_prev, sma20, sma80, slowD, slowK,rsi,dji, episode_i):

    signal = pd.Series(index=np.arange(len(data)))
    signal.fillna(value=0, inplace=True)
    signal.loc[0] = 1
    state, pdata = initializeState(data, data_prev, sma20, sma80,slowD,slowK,rsi,dji)
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
    for j in range(signal.shape[0]):
        if signal.iloc[j] < 0:
            short += 1
        elif signal.iloc[j] > 0:
            long += 1
        else:
            hold+=1
    print('Episode #: ',episode_i, ' r-Buy: ', long, ', r-Sell: ', short, 'r-hold: ',hold)

    bt = twp.Backtest(data, signal, signalType='shares')
    plt.figure(figsize=(20, 20))
    plt.subplot(2,1,1)
    plt.title("trades "+str(episode_i))
    plt.xlabel("timestamp")
    bt.plotTrades()
    plt.subplot(2, 1, 2)
    plt.title("PnL "+str(episode_i))
    plt.xlabel("timestamp")
    bt.pnl.plot(style='-')

    if episode_i % 5 == 0:
        plt.savefig('plot/' + str(episode_i) + '.png')
    plt.show()
    plt.close()
    endReward = bt.pnl.iloc[-1]

    return endReward, realProfit
