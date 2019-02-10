from matplotlib import pyplot as plt
import random

from functions import *


# get price & techinical indicator data as pandas dataframe
pdata, data = getData(1)

# load model from file
model = getModel(1)

# initialize signal(buy/sell/hold decisions)
signal = pd.Series(index=np.arange(len(data)))
signal.fillna(value=0, inplace=True)
signal.loc[0] = 1
state= initializeState(pdata)
# indicate if now it's last state
endState = 0
timeStep = 1
inventory = []
profit = 0
while not endState:

    action = (np.argmax(model.predict(state, batch_size=1)))

    # perform trade and move to next state
    nextState, timeStep, signal, endState, profit,reward = trade(action, pdata, signal, timeStep, inventory, data, profit)

    state = nextState

while len(inventory) > 0:
    profit += (data.iloc[-1] - inventory.pop(0))*10  # unsure if should be calculated this way??

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

bt = twp.Backtest(data, signal, signalType='shares')
endReward = bt.pnl.iloc[-1]
plt.figure(figsize=(20, 10))

print("profit is ", profit)

bt = twp.Backtest(data, signal, signalType='shares')
print(bt.data)

plt.figure(figsize=(20, 20))
plt.subplot(2, 1, 1)
plt.title("trades")
plt.xlabel("timestamp")
bt.plotTrades()
plt.subplot(2, 1, 2)
plt.title("PnL")
plt.xlabel("timestamp")
bt.pnl.plot(style='-')
plt.tight_layout()
plt.savefig('plot/summary_test' + '.png', bbox_inches='tight', pad_inches=1, dpi=72)
plt.show()
