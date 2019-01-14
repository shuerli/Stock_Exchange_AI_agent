import random
import timeit
from matplotlib import pyplot as plt

from functions import *

# number of training episodes
episodes = 500

# discount factor, higher gamma, more important future reward
gamma = 0.95

# probability of action randomness
epsilon = 1

# batchsize to be fed into the neural net
minibatch_size = 32

# experience replay buffer size
buffer_size = 64

# experience replay buffer iterator
replayIter = 0

# initialize experience replay buffer
replay_buffer = []

# profit and loss progress during the training
pnl_progress = []

# earning progress during the training
profit_progress = []

# profit and loss progress without random actions
real_pnl_progress = []

# earning progress without random actions
real_profit_progress = []

# get price & techinical indicator data as pandas dataframe
data, data_prev, sma20, sma80 = getData()

# get neural net model
model = getModel(0)

# initialize signal(buy/sell/hold decisions)
signal = pd.Series(index=np.arange(len(data)))
signal.fillna(value=0, inplace=True)

# timer start
startTime = timeit.default_timer()

# training start
for i in range(episodes):

    state, pdata = initializeState(data, data_prev, sma20, sma80)

    # indicate if now it's last state
    endState = 0

    timeStep = 1

    # share inventory & profit made
    inventory = []
    profit = 0

    while not endState:

        # get q values from neural net
        qValues = model.predict(state, batch_size=1)

        # exploitation vs exploration
        if random.random() < epsilon:
            action = np.random.randint(0, 3)  # random action
        else:
            action = (np.argmax(qValues))  # agent action, argmax returns index of max value

        # perform trade and move to next state
        nextState, timeStep, signal, endState, profit = trade(action, pdata, signal, timeStep, inventory, data, profit)

        # obtain reward
        reward = getReward(timeStep, signal, endState, data)

        # Experience Replay

        # fill buffer if not full
        if len(replay_buffer) < buffer_size:
            replay_buffer.append((state, action, reward, nextState, endState))
        # reinforcement learning happens here
        else:
            # start to replace from beginning if buffer is full
            replay_buffer[replayIter] = (state, action, reward, nextState, endState)
            replayIter += 1
            if replayIter == buffer_size - 1:
                replayIter = 0

            # sample random minibatch of transitions
            miniBatch = random.sample(replay_buffer, minibatch_size)

            # input and output batch to be fit the neural net
            x_train = []
            y_train = []

            for transition in miniBatch:
                state_, action_, reward_, newState_, endState_ = transition
                # reward of current state
                Q = model.predict(state, batch_size=1)
                # reward of next state
                newQ = model.predict(newState_, batch_size=1)
                newQMax = np.max(newQ)

                if not endState_:
                    # non-terminal state
                    update = (reward_ + (gamma * newQMax))
                else:
                    # terminal state
                    update = reward_

                # update q table
                Q[0][action] = update

                x_train.append(state)
                y_train.append(Q)

            # reduce dimension
            x_train = np.squeeze(np.array(x_train), axis=(1))
            # print(x_train)
            y_train = np.squeeze(np.array(y_train), axis=(1))
            # print(y_train)

            # fit data into the model
            model.fit(x_train, y_train, epochs=2, verbose=0,
                      batch_size=minibatch_size)  # minibatch_size can be <= # of x_train datas

        state = nextState

    # decrement epsilon over time
    if epsilon > 0.1:
        epsilon -= (1 / episodes)

    # calculate final cash
    while len(inventory) > 0:
        profit += data.iloc[-1] - inventory.pop(0)  # unsure if should be calculated this way??

    # print out decisions
    long = 0
    short = 0
    hold = 0
    for j in range(signal.shape[0]):
        if signal.iloc[j] < 0:
            short += 1
        elif signal.iloc[j] > 0:
            long += 1
        else:
            hold += 1
    print('Buy: ', long, ', Sell: ', short, ', Hold: ', hold)

    # incorporate all data into backtest module
    bt = twp.Backtest(data, signal, signalType='shares')

    # final pnl value
    endReward = bt.pnl.iloc[-1]

    #set plot size
    plt.figure(figsize=(20, 10))

    # test real performance of the agent without randomness
    r_reward, r_profit = test(model, data, data_prev, sma20, sma80)

    # append real performance result to the progress list
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

    # save model to file every 20 episodes
    if i % 20 == 0 and i >= episodes / 2:
        model.save('model/episode' + str(i) + '.h5')

# timer ends
time_elapsed = timeit.default_timer() - startTime
print("Finished in ", time_elapsed, "seconds")

model.save('model/FINAL_model.h5')
bt = twp.Backtest(data, signal, signalType='shares')
bt.data['delta'] = bt.data['shares'].diff().fillna(0)

print(bt.data)

# smoothing the curve using moving average

pnl_df = pd.DataFrame({'col': pnl_progress})
profit_df = pd.DataFrame({'col': profit_progress})

pnl_df = pnl_df.rolling(window=60).mean()
profit_df = profit_df.rolling(window=60).mean()

r_pnl_df = pd.DataFrame({'col': real_pnl_progress})
r_profit_df = pd.DataFrame({'col': real_profit_progress})

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
plt.plot(pnl_df, 'b')
plt.plot(r_pnl_df, 'r')
plt.subplot(4, 1, 4)
plt.title("Profit progress")
plt.xlabel("Episode(s)")
plt.plot(profit_df, 'b')
plt.plot(r_profit_df, 'r')
plt.tight_layout()
plt.savefig('plot/summary' + '.png', bbox_inches='tight', pad_inches=1, dpi=72)
plt.show()
