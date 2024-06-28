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
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor

#--------- File with events for reconstruction:
#--- evts for training:
#infile = "tankPMT_forEnergy.csv"
infile = "tankPMT_forEnergy_till800MeV.csv"
#

# Set TF random seed to improve reproducibility
seed = 150
np.random.seed(seed)

print( "--- opening file with input variables!")
#--- events for training - MC events
filein = open(str(infile))
print("evts for training in: ",filein)
Dataset = np.array(pd.read_csv(filein))
lambdas, features, rest, nhits, recoVtxFOM, TrueTrackLengthInMrd, labels = np.split(Dataset,[1100,5500,5505,5506,5507,5508],axis=1)
print("features.shape ", features.shape)
print(features[:2])
print("nhits.shape ", nhits.shape)
print(nhits[:2])
print(TrueTrackLengthInMrd[:2])
print(labels[:2])

#split events in train/test samples:
num_events, num_pixels = features.shape
print(num_events, num_pixels)
np.random.seed(0)
#train_x = features[:20000]
#train_y = labels[:20000]
train_x = features[:18000]
train_y = labels[:18000]
print("train sample features shape: ", train_x.shape," train sample label shape: ", train_y.shape)

# Scale data (training set) to 0 mean and unit standard deviation.
scaler = preprocessing.StandardScaler()
train_x = scaler.fit_transform(train_x)

def create_model():
    # create model
    model = Sequential()
    model.add(Dense(20, input_dim=4400, kernel_initializer='normal', activation='relu'))
    model.add(Dense(10, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal', activation='relu'))
    # Compile model
    model.compile(loss='mean_squared_error', optimizer='Adamax', metrics=['mean_absolute_error'])
    return model

estimator = KerasRegressor(build_fn=create_model, epochs=10, batch_size=2, verbose=0)

# checkpoint
filepath="weights_bets_till800MeV.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True, mode='auto')
callbacks_list = [checkpoint]
# Fit the model
history = estimator.fit(train_x, train_y, validation_split=0.33, epochs=10, batch_size=2, callbacks=callbacks_list, verbose=0)

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
