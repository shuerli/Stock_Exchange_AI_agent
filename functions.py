import pandas as pd
import backtest as twp
import numpy as np
import time
from sklearn import preprocessing

from keras.models import Sequential
from keras.layers.core import Dense, Dropout
from keras.layers.recurrent import LSTM
from keras.models import load_model
from keras.optimizers import Adam
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.techindicators import TechIndicators
# pnl explained : https://www.investopedia.com/ask/answers/how-do-you-calculate-percentage-gain-or-loss-investment/
# backtesting: a strategy to analyze how accurate a model did in performing trading with historycal data

def indicators(stock,time):
    ti = TechIndicators(key='H7M3SZXXJ81BCUQD', output_format='pandas', indexing_type='date')

    data, meta_data = ti.get_sma(symbol=stock, interval=time, time_period=20)
    data.to_csv('csv/sma20.csv')

    data, meta_data = ti.get_sma(symbol=stock, interval=time, time_period=80)
    data.to_csv('csv/sma80.csv')

    data, meta_data = ti.get_stoch(symbol=stock, interval=time)
    data.to_csv('csv/stoch.csv')

    data, meta_data = ti.get_rsi(symbol=stock, interval=time)
    data.to_csv('csv/rsi.csv')



def timeseries(stock,time):
    ts = TimeSeries(key='H7M3SZXXJ81BCUQD', output_format='pandas', indexing_type='date')
    data, meta_data = ts.get_intraday(symbol=stock,interval=time, outputsize='full')
    data.to_csv('csv/price.csv')

    data, meta_data = ts.get_intraday(symbol='DJI',interval=time, outputsize='full')
    data.to_csv('csv/dji.csv')




def getModel(test):

    #number of inputs
    num_inputs = 8

    if test:
        model = load_model('model/FINAL_model.h5')
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
        model.add(Dropout(0.2))
        model.add(Dense(units=32, activation="relu"))
        model.add(Dropout(0.2))
        model.add(Dense(units=8, activation="relu"))
        model.add(Dense(3, activation="linear"))
        model.compile(loss="mse", optimizer=Adam(lr=0.001))
    return model


def merge_data():
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
    final = final.drop(columns=['date'])
    return final


# read data into pandas dataframe from csv file
def getData(test):
    
    '''raw_data = merge_data()

    # pre-process data using standard scalar
    scaler = preprocessing.StandardScaler() #unit standard derivation and 0 mean
    processed_data = scaler.fit_transform(raw_data.values)

    data = pd.DataFrame(processed_data, index = raw_data.index,columns=raw_data.columns)'''
    data=merge_data()

    '''if test == 0:
        data = data[1200:1700]
    elif test == 1:
        data = data[1501:1801]
    else:
        data = data



    data = data.reset_index()'''

    price = data['price'].values
    price2 = np.diff(price)
    price2 = np.insert(price2,0,0)
    sma20 = data['sma20'].values
    sma80 = data['sma80'].values
    slowD = data['SlowD'].values
    slowK = data['SlowK'].values
    rsi = data['RSI'].values
    dji = data['dji'].values


    # stack all data into a table
    pdata = np.column_stack((price, price2, sma20, sma80, slowD, slowK, rsi, dji))
    pdata = np.nan_to_num(pdata)
    
    scaler = preprocessing.StandardScaler() #unit standard derivation and 0 mean
    pdata = scaler.fit_transform(pdata)


    

    if test == 0:
        pdata = pdata[:1500]
        data = data[:1500]
    elif test == 1:
        pdata = pdata[1500:]
        data = data[1500:]
    elif test == 2:
        pdata = pdata[len(pdata)-11:len(pdata)-1]
        data = data[len(data)-11:len(data)-1]
    else:
        pdata = pdata
        data = data

    data = data.reset_index()
    data = data['price']
    
    return pdata, data

# initialize first state
def initializeState(pdata):

    # stack all data into a table
    #pdata = np.column_stack((data, data_prev, sma20, sma80,slowD,slowK, rsi, dji))
    #pdata = np.nan_to_num(pdata)
    
    #scaler = preprocessing.StandardScaler() #unit standard derivation and 0 mean
    #pdata = scaler.fit_transform(pdata)

    # initial state is 1st row of the table

    initialState = pdata[0:1, :]

    return initialState

# reward function
def getReward(timeStep, signal, price,state, endState):

    unit_reward = 1

    if not endState:
        # net earning from previous action
        net = (price[timeStep] - price[timeStep - 1]) * signal[timeStep - 1]
    else:
        bt = twp.Backtest(price, signal, signalType='shares')
        net = bt.pnl.iloc[-1]
    rewards = 0

    #intuition reward
    if net > 0:
        rewards += unit_reward*2
    elif net < 0:
        rewards -= unit_reward/2
    else:
        #rewards -= 1 #don't encourage hold
        rewards += 0



    '''# sma reward
    if not endState:
        #sma20 = state[0,0,2]
        #sma80 = state[0,0,3]
        sma20 = state[0, 2]
        sma80 = state[0,3]
        sma_net = sma20 - sma80

        if sma_net < 0 : # short sma < long sma, down trend
            if signal[timeStep - 1] < 0:
                rewards += unit_reward
        elif sma_net > 0: #short sma > long sma, up trend
            if signal[timeStep - 1] > 0:
                rewards += unit_reward'''





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
                profit += (data[timeStep] - inventory.pop(0))*10
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
def test_agent(pdata, model, data, episode_i):

    signal = pd.Series(index=np.arange(len(data)))
    signal.fillna(value=0, inplace=True)
    signal.loc[0] = 1
    state = initializeState(pdata)
    endState = 0
    timeStep = 1
    realProfit = 0
    realInventory = []
    while not endState:
        Q = model.predict(state, batch_size=1)
        action = (np.argmax(Q))
        nextState, timeStep, signal, endState, realProfit,reward = trade(action, pdata, signal, timeStep, realInventory, data,realProfit)

        state = nextState

    while len(realInventory) > 0:
        realProfit += (data.iloc[-1] - realInventory.pop(0))*10

    long = 0
    short = 0
    hold = 0
    newdici = changedecision(signal)
    for j in range(signal.shape[0]):
        if signal.iloc[j] < 0:
            short += 1
        elif signal.iloc[j] > 0:
            long += 1
        else:
            hold+=1


    newprofitwhold = profitcal(data,newdici,5000)
    print("Profit with hold: ", newprofitwhold , "  Orignal Profit: ", realProfit)
    print('Episode #: ', episode_i, ' No   Random, Buy: ', long, ', Sell: ', short, ', Hold: ', hold)

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

    if episode_i % 10 == 0:
        plt.savefig('plot/' + str(episode_i) + '.png')
    plt.show()
    plt.close()

    bt2 = twp.Backtest(data, newdici, signalType='shares')
    plt.figure(figsize=(20, 20))
    plt.subplot(2,1,1)
    plt.title("trades "+str(episode_i))
    plt.xlabel("timestamp")
    bt2.plotTrades()
    plt.subplot(2, 1, 2)
    plt.title("PnL "+str(episode_i))
    plt.xlabel("timestamp")
    bt2.pnl.plot(style='-')

    if episode_i % 10 == 0:
        plt.savefig('plot/' + str(episode_i) + '_hold.png')

    plt.show()
    plt.close()
    endReward = bt.pnl.iloc[-1]

    return endReward, realProfit

def changedecision(decisions):
    newdeci = decisions.copy()
    prev = newdeci.iloc[0]
    for j in range(newdeci.shape[0] - 1):
        temp = newdeci.iloc[j + 1]
        if prev == newdeci.iloc[j+1]:
            newdeci.iloc[j + 1] = 0
        prev = temp
    return newdeci

def profitcal(price, decisions, initial):
    cash = initial
    share = 0
    for i in range(decisions.shape[0]-1):
        signal = decisions.iloc[i+1]
        if signal > 0 and cash != 0:
            share = cash/price[i]
            cash = 0
        elif signal < 0 and share != 0:
            cash = share * price[i]
            share = 0
    if share != 0:
        cash = share * price[i]

    return cash - initial

