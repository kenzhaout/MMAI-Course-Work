# -*- coding: utf-8 -*-
"""
Created on Mon Jun 14 14:10:34 2021

@author: ken
"""
import pandas as pd
import numpy as np
from tensorflow import keras
from tensorflow.keras.layers import Dense, Input
from keras import regularizers
from tensorflow.keras.models import Model
import random

random.seed(10)

def load_data(fname):
    data = pd.read_csv(fname, index_col=0)
    X = data.values[:, 2:].astype('float64')
    years = data['year']
    X_train = X[years < 2020.]
    X_valid = X[years == 2020.]
    tmp = data.index[data['year'] == 2020.]
    tickers = np.array([ticker.rstrip('_2020') for ticker in tmp])
    return X_train, X_valid, tickers

X_train, X_valid, tickers = load_data('C:/Users/ken/Downloads/MMAI5500_Assignment2/assign2_data.csv')
X_train = X_train.reshape(-1, 250, 1)
X_valid = X_valid.reshape(-1, 250, 1)

encoding_dim = 5 
input_shape = X_train.shape[1:]
num_stock = len(tickers)

# connect all layers
input_img = Input(shape=input_shape)
encoded = Dense(encoding_dim, activation='relu', kernel_regularizer=regularizers.l2(0.01))(input_img)
decoded = Dense(num_stock, activation= 'sigmoid', kernel_regularizer=regularizers.l2(0.01))(encoded)

# construct and compile AE model
autoencoder = Model(input_img, decoded)
autoencoder.compile(loss='mse', optimizer='adam')

autoencoder.fit(X_train, X_train, shuffle=False, epochs=500, batch_size = 10)
reconstruct = autoencoder.predict(X_valid)


communal_information = []
for i in range(0,118):
    diff = np.linalg.norm((X_valid[:,i] - reconstruct[:,i])) # 2 norm difference
    communal_information.append(float(diff))
    
ranking = np.array(communal_information).argsort()
result = []
for stock_index in ranking:
    result.append(tickers[stock_index])

most_communal = result[-5:]
least_communal = result[0:20]

print('Most communal', '-------------', (', '.join(most_communal)), 
      '\nLease communal', '-------------', (', '.join(least_communal)), sep="\n")