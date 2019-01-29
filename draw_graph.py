from matplotlib import pyplot as plt
import pandas as pd

pnl_progress = []
profit_progress = []
real_pnl_progress = []
real_profit_progress = []

# open file and read the content in a list
with open('progress_file/pnl_progress.txt', 'r') as filehandle:
    for line in filehandle:
        # remove linebreak which is the last character of the string
        currentPlace = line[:-1]

        # add item to the list
        pnl_progress.append(float(currentPlace))

# open file and read the content in a list
with open('progress_file/profit_progress.txt', 'r') as filehandle:
    for line in filehandle:
        # remove linebreak which is the last character of the string
        currentPlace = line[:-1]

        # add item to the list
        profit_progress.append(float(currentPlace))

# open file and read the content in a list
with open('progress_file/real_pnl_progress.txt', 'r') as filehandle:
    for line in filehandle:
        # remove linebreak which is the last character of the string
        currentPlace = line[:-1]

        # add item to the list
        real_pnl_progress.append(float(currentPlace))

# open file and read the content in a list
with open('progress_file/real_profit_progress.txt', 'r') as filehandle:
    for line in filehandle:
        # remove linebreak which is the last character of the string
        currentPlace = line[:-1]

        # add item to the list
        real_profit_progress.append(float(currentPlace))

print(pnl_progress)

pnl_df = pd.DataFrame({'col': real_pnl_progress})

pnl_df = pnl_df.rolling(window=400).mean()

plt.plot(pnl_df)
plt.show()