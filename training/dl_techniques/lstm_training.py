import pandas as pd
from training.dl_techniques.lstm import train_lstm
from training.dl_techniques.lstm import optimize_hyperparameters

train = pd.read_csv('/Users/yarianrango/Documents/School/Master-AI-VU/ML quantified/ML4QS/training/train.csv', index_col=0).reset_index(drop=True)

# batch = train.loc[:round(len(train)*0.3),:]

train_X = train.drop('combined', axis = 1)
train_y = train.loc[:,'combined']

train_lstm(train_X, train_y, num_epochs=100, m1=True, lstm_layers=1)

#grid_result = optimize_hyperparameters(train_X, train_y)



