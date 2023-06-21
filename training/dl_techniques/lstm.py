import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM


train = pd.read_csv('/Users/yarianrango/Documents/School/Master-AI-VU/ML quantified/ML4QS/training/train.csv', index_col=0).reset_index(drop=True)
val = pd.read_csv('/Users/yarianrango/Documents/School/Master-AI-VU/ML quantified/ML4QS/training/val.csv', index_col=0).reset_index(drop=True)
test = pd.read_csv('/Users/yarianrango/Documents/School/Master-AI-VU/ML quantified/ML4QS/training/test.csv', index_col=0).reset_index(drop=True)

scaler = MinMaxScaler(feature_range=(0, 1))
train_X = train.drop('combined', axis = 1)
train_y = train[['combined']]
train_X = scaler.fit_transform(train_X)

val_X = val.drop('combined', axis = 1)
val_y = val[['combined']]
val_X = scaler.fit_transform(val_X)

test_X = test.drop('combined', axis = 1)
test_y = test[['combined']]
test_X = scaler.fit_transform(test_X)



# model = Sequential()
#
# model.add(LSTM(50))
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#
# history = model.fit(train_X, train_y, epochs=10, batch_size=50,validation_split=0.1)


