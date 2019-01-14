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


from functions import *

episodes = 500
gamma = 0.95  # higher gamma, more important future reward
epsilon = 1
training_batch_size = 50
buffer_size = 100
replay_buffer = []
replayIter = 0
pnl_progress = []
profit_progress = []

real_pnl_progress = []
real_profit_progress = []

data,data_prev, sma20, sma80 = getData()
model = getModel(0)

signal = pd.Series(index=np.arange(len(data)))
signal.fillna(value=0, inplace=True)


startTime = timeit.default_timer()
for i in range(episodes):

    state,pdata = initializeState(data, data_prev,sma20, sma80)
    endState=0
    timeStep=1
    inventory = []
    profit = 0
    while not endState:
        qValues = model.predict(state, batch_size=1)

        #exploitation vs exploration
        if (random.random() < epsilon):       #random decision
            action = np.random.randint(0, 3)
        else:                                 #agent decision
            action = (np.argmax(qValues))

        nextState, timeStep, signal, endState, profit = trade(action,pdata,signal,timeStep,inventory,data,profit)

        reward = getReward(timeStep, signal, endState, data)

        # Experience Replay
        if len(replay_buffer) < buffer_size:
            replay_buffer.append((state, action, reward, nextState,endState))
        else:
            replay_buffer[replayIter] = (state, action, reward, nextState,endState)
            replayIter+=1
            if replayIter == buffer_size-1:
                replayIter = 0
            miniBatch = random.sample(replay_buffer, training_batch_size)

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
            model.fit(x_train,y_train,epochs=1,verbose=0,batch_size=training_batch_size) #training_batch_size can be <= # of x_train datas

        state = nextState

    # learning_progress.append((reward))
    if epsilon > 0.1:  # decrement epsilon over time
        epsilon -= (1 / episodes)

    while len(inventory) > 0:
        profit += data.iloc[-1] - inventory.pop(0) #unsure if should be calculated this way??

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
    print('Buy: ', long, ', Sell: ', short, ', Hold: ',hold)

    bt = twp.Backtest(data, signal, signalType='shares')
    endReward = bt.pnl.iloc[-1]
    plt.figure(figsize=(20, 10))


    r_reward, r_profit = test(model,data,data_prev,sma20,sma80)

    real_pnl_progress.append((r_reward))
    real_profit_progress.append((r_profit))

    pnl_progress.append((endReward))
    profit_progress.append((profit))
    print("Episode #: %s PnL:      %f Epsilon: %f Profit: %f" % (i, endReward, epsilon, profit))
    print("Episode #: %s real PnL: %f               real Profit: %f" % (i, r_reward, r_profit))
    bt.plotTrades()
 
    plt.suptitle(str(i))
    plt.savefig('plot/' + str(i) + '.png')
    plt.show()

    if i % 20 == 0:
        model.save('model/episode'+str(i)+'.h5')

elapsed = np.round(timeit.default_timer() - startTime, decimals=2)
print("Completed in %f" % (elapsed,))

model.save('model/FINAL_model.h5')
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
plt.savefig('plot/summary' + '.png', bbox_inches='tight', pad_inches=1, dpi=72)
plt.show()
