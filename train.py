import pandas as pd
import backtest as twp
import numpy as np
from matplotlib import pyplot as plt
import random, timeit
from sklearn import preprocessing

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.recurrent import LSTM
from keras.models import load_model


# pnl explained : https://www.investopedia.com/ask/answers/how-do-you-calculate-percentage-gain-or-loss-investment/
#backtesting: a strategy to analyze how accurate a model did in performing trading with historycal data
def getData():
    price = pd.read_csv('csv/bull_test/GOOG1min100.csv')
    #price = price.tail(100).reset_index()
    price = price[900:1400]
    price = price.reset_index()
    price = price['4. close']
    sma20 = pd.read_csv('csv/bull_test/GOOG1min100sma20.csv')
    #sma20 = sma20.tail(100).reset_index()
    sma20 = sma20[900:1400]
    sma20 = sma20.reset_index()
    sma20 = sma20['SMA']
    sma80 = pd.read_csv('csv/bull_test/GOOG1min100sma80.csv')
    #sma80 = sma80.tail(100).reset_index()
    sma80 = sma80[900:1400]
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
    pdata = np.expand_dims(scaler.fit_transform(pdata), axis=1) #add dimension for lstm input
    initialState = pdata[1:2,:,:]


    return initialState, pdata

def trade(action, pdata, signal, timeStep,inventory,data,totalProfit):
    profit = totalProfit
    timeStep += 1
    state = pdata[timeStep: timeStep + 1,:,:]  # preserves dimension for lstm input
    if timeStep +1 == pdata.shape[0]:
        endState = 1
    else:
        endState = 0

    if action == 1: #buy 1
        signal.loc[timeStep-1] = 1
        inventory.append(data[timeStep-1])
    elif action == 2:# and len(inventory) > 0: #sell 1
        signal.loc[timeStep-1] = -1
        if len(inventory) > 0:
            profit += data[timeStep-1] - inventory.pop(0)
    else:
        signal.loc[timeStep-1] = 0

    return state, timeStep, signal, endState, profit

def getReward(timeStep, signal, endState, price):

    net = (data[timeStep] - data[timeStep-1])*signal[timeStep-1]
    rewards = 0
    if not endState:
        
        if net >0:
            rewards = 1
        elif net < 0:
            rewards = -1

    else:
        bt = twp.Backtest(price, signal, signalType='shares')
        rewards = bt.pnl.iloc[-1]
        
    return rewards

def getModel():

    num_features = 4
    exist =  0

    if exist:
        model = load_model('model/episode400.h5')
    else:
        model = Sequential()
        model.add(LSTM(80, input_shape=(1, num_features), return_sequences=True, stateful=False))
        model.add(Dropout(0.2))
        model.add(LSTM(80, return_sequences=False, stateful=False))
        model.add(Dropout(0.2))
        model.add(Dense(3, activation="linear"))
        model.compile(optimizer='adam', loss='mse')
    return model

def test(model,data, sma20, sma80):
    # This function is used to evaluate the performance of the system each epoch, without the influence of epsilon and random actions
    signal = pd.Series(index=np.arange(len(data)))
    signal.fillna(value=0, inplace=True)

    state, pdata = initializeState(data, sma20, sma80)
    endState = 0
    timeStep = 1
    realProfit = 0
    realInventory = []
    while not endState:
        Q = model.predict(state, batch_size=1)
        action = (np.argmax(Q))
        #print(Q,'***',action)
        nextState, timeStep, signal, endState, realProfit = trade(action, pdata, signal, timeStep, realInventory, data,realProfit)

        state = nextState
    while len(realInventory) > 0:
        realProfit += data.iloc[-1] - realInventory.pop(0)

    long = 0
    short = 0
    for i in range(signal.shape[0]):
        if signal.iloc[i] < 0:
            short += 1
        elif signal.iloc[i] > 0:
            long += 1
    print('r-Buy: ',long,', r-Sell: ',short)

    bt = twp.Backtest(data, signal, signalType='shares')
    endReward = bt.pnl.iloc[-1]

    return endReward,realProfit

# Main program start

episodes = 500
gamma = 0.95  # higher gamma, more important future reward
epsilon = 1
batchSize = 50
buffer = 100
replay = []
replayIter = 0
pnl_progress = []
profit_progress = []

real_pnl_progress = []
real_profit_progress = []

data, sma20, sma80 = getData()
model = getModel()

signal = pd.Series(index=np.arange(len(data)))
signal.fillna(value=0, inplace=True)


startTime = timeit.default_timer()
for i in range(episodes):

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

        reward = getReward(timeStep, signal, endState, data)

        # Experience Replay
        if len(replay) < buffer:
            replay.append((state, action, reward, nextState,endState))
        else:
            if replayIter < (buffer - 1):
                replayIter += 1
            else:
                replayIter = 0
            replay[replayIter] = (state, action, reward, nextState,endState)
            miniBatch = random.sample(replay, batchSize)

            x_train = []
            y_train = []
            for memory in miniBatch:
                state_, action_, reward_, newState_,endState_ = memory
                Q = model.predict(state, batch_size=1)
                newQ = model.predict(newState_, batch_size=1)
                newQMax = np.max(newQ)

                if not endState_:  # non-terminal state
                    update = (reward_ + (gamma * newQMax))
                else:  # terminal state
                    update = reward_

                Q[0][action] = update

                x_train.append(state)
                y_train.append(Q)

            x_train =np.squeeze(np.array(x_train), axis=(1))
            #print(x_train)
            y_train = np.squeeze(np.array(y_train),axis=(1))
            #print(y_train)
            model.fit(x_train,y_train,epochs=1,verbose=0,batch_size=batchSize) #batchsize can be <= # of x_train datas

        state = nextState

    # learning_progress.append((reward))
    if epsilon > 0.1:  # decrement epsilon over time
        epsilon -= (1 / episodes)

    while len(inventory) > 0:
        profit += data.iloc[-1] - inventory.pop(0) #unsure if should be calculated this way??

    long = 0
    short = 0
    for j in range(signal.shape[0]):
        if signal.iloc[j] < 0:
            short += 1
        elif signal.iloc[j] > 0:
            long += 1
    print('Buy: ', long, ', Sell: ', short)

    bt = twp.Backtest(data, signal, signalType='shares')
    endReward = bt.pnl.iloc[-1]
    plt.figure(figsize=(20, 10))


    r_reward, r_profit = test(model,data,sma20,sma80)

    real_pnl_progress.append((r_reward))
    real_profit_progress.append((r_profit))

    pnl_progress.append((endReward))
    profit_progress.append((profit))
    print("Episode #: %s PnL:      %f Epsilon: %f Profit: %f" % (i, endReward, epsilon, profit))
    print("Episode #: %s real PnL: %f               real Profit: %f" % (i, r_reward, r_profit))
    bt.plotTrades()
 
    plt.suptitle(str(i))
    plt.savefig('plots/bull_test/' + str(i) + '.png')
    plt.show()

    if i % 20 == 0:
        model.save('model/bull_test/episode'+str(i)+'.h5')

elapsed = np.round(timeit.default_timer() - startTime, decimals=2)
print("Completed in %f" % (elapsed,))

model.save('model/bull_test/FINAL_model.h5')
bt = twp.Backtest(data, signal, signalType='shares')
bt.data['delta'] = bt.data['shares'].diff().fillna(0)

print(bt.data)

#smoothing the curve

pnl_df = pd.DataFrame({'col':pnl_progress})
profit_df = pd.DataFrame({'col':profit_progress})

pnl_df = pnl_df.rolling(window=60).mean()
profit_df = profit_df.rolling(window=60).mean()

r_pnl_df = pd.DataFrame({'col':real_pnl_progress})
r_profit_df = pd.DataFrame({'col':real_profit_progress})

r_pnl_df = r_pnl_df.rolling(window=60).mean()
r_profit_df = r_profit_df.rolling(window=60).mean()


plt.figure(figsize=(30, 20))
plt.subplot(4, 1, 1)
plt.title("trades")
plt.xlabel("timestamp")
bt.plotTrades()
plt.subplot(4, 1, 2)
plt.title("PnL")
plt.xlabel("timestamp")
bt.pnl.plot(style='-')
plt.subplot(4, 1, 3)
plt.title("PnL progress")
plt.xlabel("Episode(s)")
plt.plot(pnl_df,'b')
plt.plot(r_pnl_df,'r')
plt.subplot(4, 1, 4)
plt.title("Profit progress")
plt.xlabel("Episode(s)")
plt.plot(profit_df,'b')
plt.plot(r_profit_df,'r')
plt.tight_layout()
plt.savefig('plots/bull_test/summary' + '.png', bbox_inches='tight', pad_inches=1, dpi=72)
plt.show()
