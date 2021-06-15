'''
Keras DNN to predict vertex from quads (WATCHMAN) 
'''
import sys
import glob
import numpy as np
import pandas as pd
import tensorflow as tf
import tempfile
import random
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from array import array
from sklearn import datasets
from sklearn import metrics
from sklearn import model_selection
from sklearn import preprocessing
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from sklearn.preprocessing import MinMaxScaler

#--------- File with events for reconstruction:
#--- evts for training:
infile = "seeds_fixedpos.csv"
#infile = "seeds_randompos_large.csv"

# Set random seeds to improve reproducibility
from numpy.random import seed
seed(1)
import tensorflow
tensorflow.random.set_seed(1)

print( "--- opening file with input variables!")
#--- events for training - MC events
filein = open(str(infile))
print("evts for training in: ",filein)

### Loop the data lines and fillin with nan if column is missing
with open(str(infile)) as temp_f:
    # get No of columns in each line
    col_count = [ len(l.split(",")) for l in temp_f.readlines() ]
### Generate column names  (names will be 0, 1, 2, ..., maximum columns - 1)
column_names = [i for i in range(0, max(col_count))]

### Read csv
dataIN = pd.read_csv(filein, header=None, delimiter=",", names=column_names).fillna(0)
print(dataIN.head())
Dataset = np.array(dataIN)
labels, features = np.split(Dataset,[3],axis=1)
#features = np.concatenate((nhits, features0),axis=1)
print(features.shape)

in_dim = features.shape[1]
out_dim = labels.shape[1]
print("IN/OUT dim: ", in_dim, ",", out_dim)

#split events in train/test samples:
num_events, num_pixels = features.shape
print(num_events, num_pixels)
np.random.seed(0)
train_x0 = features[:25000]
train_y = labels[:25000]
print("train sample features shape: ", train_x0.shape," train sample label shape: ", train_y.shape)
print("train_y: ", train_y[:4])
#print("train_x[0] ",train_x[0])

# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
train_x = scaler.fit_transform(train_x0)


def create_model():
    # create model
    model = Sequential()
    model.add(Dense(100, input_dim=in_dim, kernel_initializer='he_uniform', activation='relu'))
#    model.add(Dense(300, kernel_initializer='he_uniform', activation='relu'))
    model.add(Dense(50, kernel_initializer='he_uniform', activation='relu'))
    model.add(Dense(out_dim, kernel_initializer='he_uniform', activation='relu'))
    # Compile model
    model.compile(loss='mean_squared_error', optimizer='Adamax', metrics=['accuracy'])
    return model
'''
    ###model used for the fixed positon:
     # create model
     model = Sequential()
     model.add(Dense(100, input_dim=in_dim, kernel_initializer='he_uniform', activation='relu'))
     model.add(Dense(50, kernel_initializer='he_uniform', activation='relu'))
     model.add(Dense(out_dim, kernel_initializer='he_uniform', activation='relu'))
     # Compile model
     model.compile(loss='mean_squared_error', optimizer='Adamax', metrics=['accuracy'])
     return model
'''
estimator = KerasRegressor(build_fn=create_model, epochs=20, batch_size=2, verbose=0)

# checkpoint
filepath="weights_bets.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True, mode='auto')
callbacks_list = [checkpoint]
# Fit the model
history = estimator.fit(train_x, train_y, validation_split=0.33, epochs=20, batch_size=2, callbacks=callbacks_list, verbose=0)
#history = estimator.fit(train_x, train_y, validation_split=0.33, callbacks=callbacks_list, verbose=0)

#-----------------------------
# summarize history for loss
f, ax2 = plt.subplots(1,1)
ax2.plot(history.history['loss'])
ax2.plot(history.history['val_loss'])
ax2.set_title('model loss')
ax2.set_ylabel('Performance')
ax2.set_xlabel('Epochs')
#ax2.set_xlim(0.,10.)
ax2.legend(['train', 'test'], loc='upper left')
plt.savefig("keras_train_test.pdf")


