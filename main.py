#!/usr/bin/env python
# coding: utf-8
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error

import keras
from keras.layers import Dense
from keras.models import Sequential
from keras.utils import to_categorical
from keras.optimizers import SGD 
from keras.callbacks import EarlyStopping
from keras.utils import np_utils
import itertools
from keras.layers import LSTM
from keras.layers.convolutional import Conv1D,Conv2D
from keras.layers.convolutional import MaxPooling1D,MaxPooling2D
from keras.layers import Dropout
from keras.layers import Bidirectional
from keras.layers import TimeDistributed, Dense, Flatten,ConvLSTM2D,RepeatVector
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import tensorflow as tf 
from keras import backend as k 


from sklearn.preprocessing import MinMaxScaler

# Allow GPU growth
config = tf.ConfigProto()

# Don't pre-allocate memory; allocate as-needed
config.gpu_options.allow_growth = True

# Create a session with the above options specified.
k.tensorflow_backend.set_session(tf.Session(config=config))


def parse(x):
	return datetime.strptime(x, '%Y %m %d %H')



def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	"""
	Frame a time series as a supervised learning dataset.
	Arguments:
		data: Sequence of observations as a list or NumPy array.
		n_in: Number of lag observations as input (X).
		n_out: Number of observations as output (y).
		dropnan: Boolean whether or not to drop rows with NaN values.
	Returns:
		Pandas DataFrame of series framed for supervised learning.
	"""
	n_vars = 1 if type(data) is list else data.shape[1]
	df = pd.DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = pd.concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg


def get_sample_by_time(df, t_type):
	#Resample data to hour (original data is minute)
	if t_type == 'h':
		return df.resample('h').mean()
	#Resample data to daily
	if t_type == 'D':
		return df.resample('D').mean()
	#Resample data to weekly
	if t_type == 'W':
		return df.resample('W').mean()
	#Resample data to minutely
	if t_type == 'm':
		return df  
		
def bi_lstm_model():
	model = Sequential()
	model.add(TimeDistributed(Conv1D(filters=64, kernel_size=2, activation='relu'), input_shape=(None, n_steps, n_features)))
	model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
	model.add(TimeDistributed(Conv1D(filters=64, kernel_size=2,activation='relu')))
	model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
	model.add(TimeDistributed(Flatten()))
	model.add(Bidirectional(LSTM(64, activation='relu',return_sequences=True)))
	model.add(Bidirectional(LSTM(64, activation='relu')))
	model.add(Dense(128))
	model.add(Dropout(0.5))
	model.add(Dense(1))
	model.compile(optimizer='adam', loss='mse',metrics = ['mae','mse','mape'])
	return model 


def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def calculate_metrics(inv_y,inv_yhat):
	# calculate metrics
	rmse = np.sqrt(mean_squared_error(inv_y, inv_yhat))
	mse = mean_squared_error(inv_y, inv_yhat)
	mae = mean_absolute_error(inv_y, inv_yhat)
	mape = mean_absolute_percentage_error(inv_y, inv_yhat)
	return rmse, mse, mae, mape 


if __name__ == '__main__':

	df = pd.read_csv("household_power_consumption.txt",delimiter=";",
					parse_dates={'dt' : ['Date', 'Time']}, infer_datetime_format=True,low_memory=False, na_values=['nan','?'], index_col='dt')
	print("Data size:", df.shape)

	#Number of missing data
	df.isnull().sum(axis=0)
	df = df.dropna(axis=0)
	print(df.head())

	df_sample = get_sample_by_time(df,"W")
	
	#Preprocessing data
	values = df_sample.values
	values = values.astype('float32')
	scaler = MinMaxScaler(feature_range=(0, 1))
	scaled = scaler.fit_transform(values)
	reframed = series_to_supervised(values, 1, 1)
	reframed.drop(reframed.columns[[8,9,10,11,12,13]], axis=1, inplace=True)
	reframed.head()

	# split into train and test sets
	values = reframed.values
	# Take 3 year for training, the last 1 year to testing
	n_train_time = 365*3
	train = values[:n_train_time, :]
	test = values[n_train_time:, :]
	# split into input and outputs
	train_X, train_y = train[:, :-1], train[:, -1]
	test_X, test_y = test[:, :-1], test[:, -1]
	# reshape input to be 3D [samples, timesteps, features]
	train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
	test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
	print(train_X.shape, train_y.shape, test_X.shape, test_y.shape) 
	# We reshaped the input into the 3D format as expected by LSTMs, namely [samples, timesteps, features].


	#CNN-Bi-LSTM 
	n_steps, n_features = train_X.shape[1], train_X.shape[2]
	n_features = 1
	n_seq = 1
	n_steps = 7

	train_X= train_X.reshape((train_X.shape[0], n_seq, n_steps, n_features))
	test_X = test_X.reshape((test_X.shape[0], n_seq, n_steps, n_features))

	#Define Model
	model = bi_lstm_model()
	print(model.summary())


	history = model.fit(train_X, train_y, epochs=100, batch_size=30, validation_data=(test_X, test_y), verbose=1)

	# summarize history for loss
	plt.plot(history.history['loss'])
	plt.plot(history.history['val_loss'])
	plt.title('Model Loss')
	plt.ylabel('Loss')
	plt.xlabel('Epoch')
	plt.legend(['Train', 'Validation'], loc='upper right')
	plt.show()

	# make a prediction
	yhat = model.predict(test_X)
	test_X = test_X.reshape((test_X.shape[0], 7))
	# invert scaling for forecast
	inv_yhat = np.concatenate((yhat, test_X[:, -6:]), axis=1)
	inv_yhat = scaler.inverse_transform(inv_yhat)
	inv_yhat = inv_yhat[:,0]
	# invert scaling for actual
	test_y = test_y.reshape((len(test_y), 1))
	inv_y = np.concatenate((test_y, test_X[:, -6:]), axis=1)
	inv_y = scaler.inverse_transform(inv_y)
	inv_y = inv_y[:,0] 


	rmse, mse, mae, mape = calculate_metrics(inv_y, inv_yhat)

	print("Test MSE: %.3f"%mse)
	print('Test RMSE: %.3f' % rmse)
	print("Test MAE: %.3f" % mae)
	print("Test MAPE: %.3f" % mape)

