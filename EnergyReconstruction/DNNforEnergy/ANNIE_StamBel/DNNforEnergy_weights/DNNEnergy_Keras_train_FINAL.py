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
from sklearn.utils import shuffle
#--------- File with events for reconstruction:
#--- evts for training:
#infile = "tankPMT_forEnergy.csv"
infile = "shuffled_cut_data.csv"
#

# Set TF random seed to improve reproducibility
seed = 150
np.random.seed(seed)

print( "--- opening file with input variables!")
#--- events for training - MC events
filein = open(str(infile))
print("evts for training in: ",filein)
df=pd.read_csv(filein)
df = df.iloc[:, 1:]
Dataset = np.array(df)
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
train_x = features[:20000]
train_y = labels[:20000]
test_x = features[20000:]
test_y = labels[20000:]
print("train sample features shape: ", train_x.shape," train sample label shape: ", train_y.shape)

# Scale data (training set) to 0 mean and unit standard deviation.
scaler = preprocessing.StandardScaler()
train_x = scaler.fit_transform(train_x)
x_transformed = scaler.transform(test_x)

#------------------
#Weight the events per energy bin to provide a flat spectrum response: 
# Create bins for target values
n_bins = 90 #300
#bins = np.linspace(y_train.min(), y_train.max(), num_bins)
bins = np.linspace(100, 1000, n_bins + 1)

# Digitize the target values to determine which bin they belong to
binned_y = np.digitize(train_y.reshape(-1), bins)

# Calculate frequencies of each bin
bin_counts = pd.Series(binned_y).value_counts().sort_index()

# Calculate weights: inverse of the bin frequency
weights = 1.0 / bin_counts[binned_y]

# Normalize weights
weights = weights / weights.mean()

#check the weighting:
plt.figure()
plt.hist(train_y, bins=bins, alpha=0.6, color='#1f77b4', edgecolor='black', label='unweighted')
plt.hist(train_y, bins=bins, weights=weights, alpha=0.4, color='#ff7f0e', edgecolor='black', label='weighted')
plt.legend()
plt.xlim(100, 1000)
plt.xlabel('E MeV')
plt.ylabel('# of events')
plt.title('Energy')
plt.savefig("energy.png")
#------------------

# create model
model = Sequential()
model.add(Dense(20, input_dim=4400, kernel_initializer='normal', activation='leaky_relu'))
model.add(Dense(10, kernel_initializer='normal', activation='leaky_relu'))
model.add(Dense(1, kernel_initializer='normal', activation='leaky_relu'))

# Compile model
model.compile(loss='mean_absolute_error', optimizer='Adamax', metrics=['mean_absolute_error'], weighted_metrics=['mse'])
    
#estimator = KerasRegressor(build_fn=create_model, epochs=10, batch_size=5, verbose=0)
"""
# checkpoint
filepath="weights_bets_after_phase_till1000w.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True, mode='auto')
callbacks_list = [checkpoint]
"""
# Fit the model
#history = estimator.fit(train_x, train_y, validation_split=0.3, epochs=10, batch_size=5, verbose=0, sample_weight=weights) #callbacks=callbacks_list,

# Compile the model
#model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mse'], weighted_metrics=['mse'])

# Train the model
history = model.fit(train_x, train_y, epochs=10, validation_split=0.3, shuffle=True, batch_size=6, sample_weight=weights)

## Evaluate the model
loss = model.evaluate(x_transformed, test_y)
print("Predictions?: ",model.predict(x_transformed)," y_test ",test_y)
print(f"Test Loss: {loss}")

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
plt.savefig("keras_train_test_till1000w.pdf")

# Save the entire model (architecture, weights, optimizer state)
model.save('model.h5')
print("Model saved to 'model.h5'")
# Save the scaler for use in prediction script:
import joblib
joblib.dump(scaler, 'scaler.pkl')
